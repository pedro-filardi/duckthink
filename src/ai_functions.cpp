//===----------------------------------------------------------------------===//
// ai_functions.cpp — the ASK text-to-SQL function plus the plumbing it needs:
// provider secrets, the model registry, LLM transport, and call metrics. The
// parser rewrites the `ASK ... ON ... RETURN ...` keyword form to ask(); the
// transport lives in provider.cpp; shared decls come from duckthink_internal.hpp.
//===----------------------------------------------------------------------===//
#include "duckthink_internal.hpp"
#include "provider.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/secret/secret.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/catalog/catalog_transaction.hpp"
#include "duckdb/execution/expression_executor_state.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "yyjson.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <cstdlib>
#include <map>
#include <memory>
#include <unordered_map>

#include <sys/stat.h>

namespace yj = duckdb_yyjson;

namespace duckdb {
//===--------------------------------------------------------------------===//
// LLM: secrets, model registry, and the llm()/create_model() functions
//===--------------------------------------------------------------------===//

// Every LLM provider stores the same shape of secret: an optional API key and an
// optional endpoint override. One create-function backs all of them; adding a
// provider is just adding its name to kLlmProviders below.
static unique_ptr<BaseSecret> CreateLlmSecret(ClientContext &, CreateSecretInput &input) {
	auto secret = make_uniq<KeyValueSecret>(input.scope, input.type, input.provider, input.name);
	secret->TrySetValue("api_key", input);
	secret->TrySetValue("endpoint", input);
	secret->redact_keys = {"api_key"};
	return std::move(secret);
}

static const char *const kLlmProviders[] = {"openai", "ollama", "anthropic"};

static void RegisterLlmSecretType(ExtensionLoader &loader, const string &provider) {
	SecretType type;
	type.name = provider;
	type.deserializer = KeyValueSecret::Deserialize<KeyValueSecret>;
	type.default_provider = "config";
	type.extension = "duckthink";
	loader.RegisterSecretType(type);

	CreateSecretFunction fun;
	fun.secret_type = provider;
	fun.provider = "config";
	fun.function = CreateLlmSecret;
	fun.named_parameters["api_key"] = LogicalType::VARCHAR;
	fun.named_parameters["endpoint"] = LogicalType::VARCHAR;
	loader.RegisterFunction(fun);
}

// Resolve a declared model name into a fully-populated call: model_id + provider
// from the registry, api_key + endpoint from the matching DuckDB secret (if any).
static duckthink::LlmCall ResolveModelCall(ClientContext &context, const string &model_name) {
	duckthink::LlmModel model;
	if (!duckthink::LookupModel(model_name, model)) {
		throw InvalidInputException("llm: model '%s' is not declared. Create it first, e.g. "
		                            "CREATE MODEL('%s', 'llama3.2', 'ollama');",
		                            model_name, model_name);
	}

	duckthink::LlmCall call;
	call.provider = model.provider;
	call.model_id = model.model_id;

	// Look up a secret whose type matches the provider (openai/ollama/...).
	auto &secret_manager = SecretManager::Get(context);
	auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);
	auto match = secret_manager.LookupSecret(transaction, model.provider, model.provider);
	if (match.HasMatch()) {
		auto &kv = dynamic_cast<const KeyValueSecret &>(match.GetSecret());
		Value v;
		if (kv.TryGetValue("api_key", v) && !v.IsNull()) {
			call.api_key = v.ToString();
		}
		if (kv.TryGetValue("endpoint", v) && !v.IsNull()) {
			call.base_url = v.ToString();
		}
	}
	if (call.base_url.empty()) {
		call.base_url = duckthink::DefaultBaseUrl(model.provider);
	}
	// How long (seconds) to keep waiting out 429 rate limits before giving up.
	Value rlw;
	if (context.TryGetCurrentSetting("llm_rate_limit_wait", rlw) && !rlw.IsNull()) {
		call.rate_limit_wait_seconds = (int)rlw.GetValue<int64_t>();
	}
	// Response token cap (required by Anthropic; ignored by OpenAI/Ollama).
	Value mt;
	if (context.TryGetCurrentSetting("llm_max_tokens", mt) && !mt.IsNull()) {
		call.max_tokens = (int)mt.GetValue<int64_t>();
	}
	// Per-call stderr trace (SET duckthink_log=true).
	Value lg;
	if (context.TryGetCurrentSetting("duckthink_log", lg) && !lg.IsNull()) {
		call.log = lg.GetValue<bool>();
	}
	return call;
}

// create_model(name, model_id, provider [, if_not_exists]) -> VARCHAR status.
// Backs CREATE [OR REPLACE] MODEL(...) [IF NOT EXISTS]. Declaring an existing name
// replaces it, unless if_not_exists is true, in which case the existing model is
// kept untouched.
static void CreateModelFun(DataChunk &args, ExpressionState &, Vector &result) {
	const bool has_ine = args.ColumnCount() >= 4;
	for (idx_t r = 0; r < args.size(); r++) {
		Value name_v = args.GetValue(0, r);
		Value id_v = args.GetValue(1, r);
		Value prov_v = args.GetValue(2, r);
		if (name_v.IsNull() || id_v.IsNull() || prov_v.IsNull()) {
			throw InvalidInputException("CREATE MODEL: name, model_id and provider must all be non-NULL");
		}
		bool if_not_exists = false;
		if (has_ine) {
			Value ine_v = args.GetValue(3, r);
			if_not_exists = !ine_v.IsNull() && ine_v.GetValue<bool>();
		}
		duckthink::LlmModel existing;
		if (if_not_exists && duckthink::LookupModel(name_v.ToString(), existing)) {
			result.SetValue(r, Value("model '" + name_v.ToString() + "' already exists, kept"));
			continue;
		}
		duckthink::LlmModel model {name_v.ToString(), id_v.ToString(), StringUtil::Lower(prov_v.ToString())};
		duckthink::RegisterModel(model);
		result.SetValue(r, Value("model '" + model.name + "' -> " + model.provider + ":" + model.model_id));
	}
}

// drop_model(name) -> VARCHAR status. Backs DROP MODEL <name>.
static void DropModelFun(DataChunk &args, ExpressionState &, Vector &result) {
	for (idx_t r = 0; r < args.size(); r++) {
		Value name_v = args.GetValue(0, r);
		if (name_v.IsNull()) {
			throw InvalidInputException("DROP MODEL: name must be non-NULL");
		}
		bool dropped = duckthink::DropModel(name_v.ToString());
		result.SetValue(r, Value(dropped ? ("dropped model '" + name_v.ToString() + "'")
		                                 : ("model '" + name_v.ToString() + "' did not exist")));
	}
}

// llm_models() -> (name, model_id, provider): list the declared models.
struct LlmModelsBindData : public TableFunctionData {
	std::vector<duckthink::LlmModel> models;
};
struct LlmModelsGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
};
static unique_ptr<FunctionData> LlmModelsBind(ClientContext &, TableFunctionBindInput &,
                                              vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<LlmModelsBindData>();
	result->models = duckthink::AllModels();
	names = {"name", "model_id", "provider"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR};
	return std::move(result);
}
static unique_ptr<GlobalTableFunctionState> LlmModelsInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<LlmModelsGlobalState>();
}
static void LlmModelsScan(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->Cast<LlmModelsBindData>();
	auto &gstate = data.global_state->Cast<LlmModelsGlobalState>();
	idx_t count = 0;
	while (gstate.offset < bind.models.size() && count < STANDARD_VECTOR_SIZE) {
		auto &m = bind.models[gstate.offset++];
		output.SetValue(0, count, Value(m.name));
		output.SetValue(1, count, Value(m.model_id));
		output.SetValue(2, count, Value(m.provider));
		count++;
	}
	output.SetCardinality(count);
}

// duckthink_metrics() -> one row per LLM model + one 'embed' row. `calls` is API
// calls for llm rows and texts embedded for the embed row (which carries 0 tokens
// — local embedding is free). Compare `calls` to the rows you fed a function to
// see batching's effect; `cache_hits` shows the dedup cache's.
struct MetricsBindData : public TableFunctionData {
	std::vector<std::vector<Value>> rows;
};
static unique_ptr<FunctionData> MetricsBind(ClientContext &, TableFunctionBindInput &,
                                            vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<MetricsBindData>();
	names = {"scope",         "model",  "calls",   "cache_hits",     "input_tokens",
	         "output_tokens", "wall_s", "retries", "rate_limited_s", "errors"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT,
	                LogicalType::BIGINT,  LogicalType::BIGINT,  LogicalType::DOUBLE, LogicalType::BIGINT,
	                LogicalType::DOUBLE,  LogicalType::BIGINT};
	for (auto &s : duckthink::SnapshotLlmStats()) {
		result->rows.push_back({Value("llm"), Value(s.model), Value::BIGINT((int64_t)s.api_calls),
		                        Value::BIGINT((int64_t)s.cache_hits), Value::BIGINT((int64_t)s.input_tokens),
		                        Value::BIGINT((int64_t)s.output_tokens), Value::DOUBLE((double)s.wall_ms / 1000.0),
		                        Value::BIGINT((int64_t)s.retries), Value::DOUBLE((double)s.rate_limited_ms / 1000.0),
		                        Value::BIGINT((int64_t)s.errors)});
	}
	auto e = duckthink::SnapshotEmbedStats();
	if (e.texts > 0) {
		result->rows.push_back({Value("embed"), Value(LogicalType::VARCHAR), Value::BIGINT((int64_t)e.texts),
		                        Value::BIGINT(0), Value::BIGINT(0), Value::BIGINT(0),
		                        Value::DOUBLE((double)e.wall_ms / 1000.0), Value::BIGINT(0), Value::DOUBLE(0.0),
		                        Value::BIGINT(0)});
	}
	return std::move(result);
}
static void MetricsScan(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->Cast<MetricsBindData>();
	auto &gstate = data.global_state->Cast<LlmModelsGlobalState>();
	idx_t count = 0;
	while (gstate.offset < bind.rows.size() && count < STANDARD_VECTOR_SIZE) {
		auto &row = bind.rows[gstate.offset++];
		for (idx_t c = 0; c < row.size(); c++) {
			output.SetValue(c, count, row[c]);
		}
		count++;
	}
	output.SetCardinality(count);
}

// duckthink_metrics_reset() -> zero the counters (measure a single query).
static void MetricsResetFun(DataChunk &args, ExpressionState &, Vector &result) {
	duckthink::ResetStats();
	for (idx_t r = 0; r < args.size(); r++) {
		result.SetValue(r, Value("metrics reset"));
	}
}

//===--------------------------------------------------------------------===//
// Table function: llm_join(left_query, right_query, prompt := ..., ...)
//   The backing function for `[LEFT] LLM JOIN`. Embeddings rank + pre-filter
//   candidates (the cheap SEMANTIC JOIN model); the LLM adjudicates the gray
//   zone. argmax -> one "pick best of top-K" call per left row; all -> one
//   yes/no call per candidate pair.
//===--------------------------------------------------------------------===//
// The registered model ASK uses to generate SQL (SET ask_model='<name>').
static constexpr const char *kLlmModelSetting = "ask_model";
static constexpr const char *kLlmRateLimitWaitSetting = "llm_rate_limit_wait";

// Chat completion memoized process-wide on (model, endpoint, prompt). Temperature
// is 0 so identical requests are deterministic: calling ai_extract twice with the
// same text+instruction+type costs one API call, not two.
static string ExtractChat(const duckthink::LlmCall &call, const string &sys, const string &user) {
	static std::mutex mu;
	static std::unordered_map<string, string> cache;
	string key = call.model_id + "\x1f" + call.base_url + "\x1f" + sys + "\x1f" + user;
	{
		std::lock_guard<std::mutex> lk(mu);
		auto it = cache.find(key);
		if (it != cache.end()) {
			duckthink::RecordCacheHit(call.provider, call.model_id); // saved an API call
			return it->second;
		}
	}
	string resp = duckthink::LlmChat(call, sys, user);
	{
		std::lock_guard<std::mutex> lk(mu);
		if (cache.size() < 100000) {
			cache.emplace(std::move(key), resp);
		}
	}
	return resp;
}

static string TrimStr(const string &s) {
	size_t a = s.find_first_not_of(" \t\r\n");
	if (a == string::npos) {
		return "";
	}
	return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

//===--------------------------------------------------------------------===//
// ASK / ask_sql — natural language -> SQL over the named tables.
//   ASK('question') ON (t1, t2) RETURN (col type, ...)   [table function]
//   ask_sql(...)                                          returns the SQL only
// The parser rewrites the keyword form to ask('q','t1,t2','col type,...').
//===--------------------------------------------------------------------===//
namespace {

// Split on top-level `delim`, ignoring delimiters inside () so DECIMAL(10,2)
// survives in a RETURN schema. Always includes the final piece.
static vector<string> SplitTopLevel(const string &s, char delim) {
	vector<string> out;
	int depth = 0;
	string cur;
	for (char c : s) {
		if (c == '(') {
			depth++;
		} else if (c == ')') {
			depth = depth > 0 ? depth - 1 : 0;
		}
		if (c == delim && depth == 0) {
			out.push_back(cur);
			cur.clear();
		} else {
			cur += c;
		}
	}
	out.push_back(cur);
	return out;
}

// Single-quote a string literal for embedding in an introspection query.
static string SqlLit(const string &s) {
	string o = "'";
	for (char c : s) {
		if (c == '\'') {
			o += "''";
		} else {
			o += c;
		}
	}
	o += "'";
	return o;
}

// Parse a RETURN schema ("country TEXT, revenue DECIMAL(10,2)") into columns.
static void ParseReturnSchema(ClientContext &context, const string &schema, vector<string> &names,
                              vector<LogicalType> &types) {
	for (auto &part : SplitTopLevel(schema, ',')) {
		string col = TrimStr(part);
		if (col.empty()) {
			continue;
		}
		size_t sp = col.find_first_of(" \t");
		if (sp == string::npos) {
			throw BinderException("ASK: RETURN column '%s' needs a name and a type, e.g. 'revenue DECIMAL'", col);
		}
		string name = TrimStr(col.substr(0, sp));
		string typestr = TrimStr(col.substr(sp + 1));
		try {
			types.push_back(TransformStringToLogicalType(typestr, context));
		} catch (std::exception &) {
			throw BinderException("ASK: invalid type '%s' in RETURN(...)", typestr);
		}
		names.push_back(name);
	}
	if (names.empty()) {
		throw BinderException("ASK: RETURN (...) must declare at least one column");
	}
}

// Strip a leading ```/```sql fence and trailing ``` if the model wrapped the SQL.
static string StripSqlFences(string s) {
	s = TrimStr(s);
	if (s.rfind("```", 0) == 0) {
		size_t nl = s.find('\n');
		if (nl != string::npos) {
			s = s.substr(nl + 1);
		}
		size_t last = s.rfind("```");
		if (last != string::npos) {
			s = s.substr(0, last);
		}
	}
	return TrimStr(s);
}

// Enforce that a generated query is a single read-only statement, using DuckDB's
// own parser (not string heuristics): it must parse to exactly one statement of
// type SELECT (a `WITH ... SELECT` is a SELECT statement; anything that inserts,
// updates, deletes, drops, or chains statements is a different type and rejected).
// Returns the cleaned SQL via `cleaned`, or false with a `why`.
static bool IsReadOnlySelect(const string &sql_in, string &cleaned, string &why) {
	string sql = TrimStr(sql_in);
	if (sql.empty()) {
		why = "is empty";
		return false;
	}
	try {
		Parser parser;
		parser.ParseQuery(sql);
		if (parser.statements.size() != 1) {
			why = "must be exactly one statement";
			return false;
		}
		if (parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
			why = "is not a read-only SELECT query";
			return false;
		}
	} catch (std::exception &e) {
		why = string("is not valid SQL (") + e.what() + ")";
		return false;
	}
	if (sql.back() == ';') { // safe to strip: parsed as a single statement
		sql.pop_back();
		sql = TrimStr(sql);
	}
	cleaned = sql;
	return true;
}

// Semantic context read from a dbt target/manifest.json: per-table/column
// descriptions + synonyms, metric definitions, and join keys (from relationships
// tests). All best-effort — dbt's schema varies across versions, so missing
// fields are simply skipped.
struct DbtInfo {
	bool loaded = false;
	std::map<string, string> table_desc;                 // relation(lower) -> description
	std::map<string, std::map<string, string>> col_desc; // relation -> col(lower) -> "desc (aka ...)"
	string semantic_block;                               // dbt Semantic Layer: models/entities/dims/measures
	string metrics_block;
	string relationships_block;
};

static string ReadWholeFile(const string &path) {
	std::ifstream f(path, std::ios::binary);
	if (!f) {
		return "";
	}
	std::ostringstream ss;
	ss << f.rdbuf();
	return ss.str();
}

// A real dbt manifest.json is tens of MB; re-reading and re-parsing it on every
// ASK() call is pathological. Parse once and cache the doc process-wide, keyed by
// (path, mtime) — reparse only when the file actually changes (e.g. `dbt build`).
static std::mutex g_manifest_mu;
static std::map<string, std::pair<int64_t, yj::yyjson_doc *>> g_manifest_cache;

static yj::yyjson_doc *GetCachedManifestDoc(const string &path) {
	struct stat st;
	if (stat(path.c_str(), &st) != 0) {
		return nullptr;
	}
	int64_t mtime = (int64_t)st.st_mtime;
	std::lock_guard<std::mutex> lk(g_manifest_mu);
	auto it = g_manifest_cache.find(path);
	if (it != g_manifest_cache.end() && it->second.first == mtime) {
		return it->second.second; // fresh hit: no read, no parse
	}
	string content = ReadWholeFile(path);
	yj::yyjson_doc *doc = content.empty() ? nullptr : yj::yyjson_read(content.c_str(), content.size(), 0);
	if (it != g_manifest_cache.end()) {
		if (it->second.second) {
			yj::yyjson_doc_free(it->second.second); // retire the stale parse
		}
		it->second = {mtime, doc};
	} else {
		g_manifest_cache.emplace(path, std::make_pair(mtime, doc));
	}
	return doc;
}

// Pull the table name out of a dbt `ref('x')` / `ref('proj','x')` / `source(...)`
// string: the last single-quoted token.
static string ExtractRefName(const string &s) {
	auto r = s.rfind('\'');
	if (r == string::npos || r == 0) {
		return "";
	}
	auto l = s.rfind('\'', r - 1);
	if (l == string::npos) {
		return "";
	}
	return s.substr(l + 1, r - l - 1);
}

static DbtInfo LoadDbtInfo(const string &manifest_path, const std::set<string> &scope) {
	DbtInfo info;
	yj::yyjson_doc *doc = GetCachedManifestDoc(manifest_path); // parsed once per (path, mtime)
	if (!doc) {
		return info;
	}
	yj::yyjson_val *root = yj::yyjson_doc_get_root(doc);
	if (!root || !yj::yyjson_is_obj(root)) {
		return info; // the cache owns the doc; never freed here
	}
	auto gstr = [](yj::yyjson_val *o, const char *k) -> string {
		if (!o) {
			return "";
		}
		yj::yyjson_val *v = yj::yyjson_obj_get(o, k);
		return (v && yj::yyjson_is_str(v)) ? string(yj::yyjson_get_str(v)) : string();
	};
	auto relation_of = [&](yj::yyjson_val *node) -> string {
		string a = gstr(node, "alias");
		if (!a.empty()) {
			return StringUtil::Lower(a);
		}
		string ident = gstr(node, "identifier"); // sources
		if (!ident.empty()) {
			return StringUtil::Lower(ident);
		}
		return StringUtil::Lower(gstr(node, "name"));
	};

	std::map<string, string> id_relation; // node id -> relation, to resolve relationships

	for (const char *section : {"nodes", "sources"}) {
		yj::yyjson_val *sec = yj::yyjson_obj_get(root, section);
		if (!sec || !yj::yyjson_is_obj(sec)) {
			continue;
		}
		yj::yyjson_obj_iter it;
		yj::yyjson_obj_iter_init(sec, &it);
		yj::yyjson_val *key;
		while ((key = yj::yyjson_obj_iter_next(&it))) {
			yj::yyjson_val *node = yj::yyjson_obj_iter_get_val(key);
			string rt = gstr(node, "resource_type");
			string rel = relation_of(node);
			if (rt == "model" || rt == "seed" || rt == "snapshot" || rt == "source") {
				id_relation[string(yj::yyjson_get_str(key))] = rel;
			}
			if (scope.count(rel) == 0) {
				continue;
			}
			string d = gstr(node, "description");
			if (!d.empty()) {
				info.table_desc[rel] = d;
			}
			yj::yyjson_val *cols = yj::yyjson_obj_get(node, "columns");
			if (cols && yj::yyjson_is_obj(cols)) {
				yj::yyjson_obj_iter cit;
				yj::yyjson_obj_iter_init(cols, &cit);
				yj::yyjson_val *ck;
				while ((ck = yj::yyjson_obj_iter_next(&cit))) {
					yj::yyjson_val *col = yj::yyjson_obj_iter_get_val(ck);
					string cname = StringUtil::Lower(string(yj::yyjson_get_str(ck)));
					string full = gstr(col, "description");
					yj::yyjson_val *meta = yj::yyjson_obj_get(col, "meta");
					yj::yyjson_val *sarr = meta ? yj::yyjson_obj_get(meta, "synonyms") : nullptr;
					if (sarr && yj::yyjson_is_arr(sarr)) {
						string syn;
						size_t n = yj::yyjson_arr_size(sarr);
						for (size_t i = 0; i < n; i++) {
							yj::yyjson_val *e = yj::yyjson_arr_get(sarr, i);
							if (e && yj::yyjson_is_str(e)) {
								syn += (syn.empty() ? "" : ", ") + string(yj::yyjson_get_str(e));
							}
						}
						if (!syn.empty()) {
							full += (full.empty() ? "" : " ") + ("(aka " + syn + ")");
						}
					}
					if (!full.empty()) {
						info.col_desc[rel][cname] = full;
					}
				}
			}
		}
	}

	// dbt Semantic Layer (MetricFlow). semantic_models carry each model's entities
	// (join keys), dimensions (group-by columns) and measures (aggregatable facts).
	// We surface the in-scope models verbatim and remember every measure's owning
	// relation so metrics can be scoped below.
	std::map<string, string> measure_rel; // measure(lower) -> relation(lower)
	auto join_arr = [&](yj::yyjson_val *arr, const std::function<string(yj::yyjson_val *)> &fmt) {
		string s;
		if (!arr || !yj::yyjson_is_arr(arr)) {
			return s;
		}
		size_t n = yj::yyjson_arr_size(arr);
		for (size_t i = 0; i < n; i++) {
			string e = fmt(yj::yyjson_arr_get(arr, i));
			if (!e.empty()) {
				s += (s.empty() ? "" : ", ") + e;
			}
		}
		return s;
	};
	yj::yyjson_val *sms = yj::yyjson_obj_get(root, "semantic_models");
	if (sms && yj::yyjson_is_obj(sms)) {
		string block;
		yj::yyjson_obj_iter it;
		yj::yyjson_obj_iter_init(sms, &it);
		yj::yyjson_val *key;
		while ((key = yj::yyjson_obj_iter_next(&it))) {
			yj::yyjson_val *sm = yj::yyjson_obj_iter_get_val(key);
			yj::yyjson_val *nr = yj::yyjson_obj_get(sm, "node_relation");
			string rel = nr ? StringUtil::Lower(gstr(nr, "alias")) : string();
			string smname = gstr(sm, "name");
			yj::yyjson_val *meas = yj::yyjson_obj_get(sm, "measures");
			if (meas && yj::yyjson_is_arr(meas) && !rel.empty()) {
				size_t n = yj::yyjson_arr_size(meas);
				for (size_t i = 0; i < n; i++) {
					string mn = gstr(yj::yyjson_arr_get(meas, i), "name");
					if (!mn.empty()) {
						measure_rel[StringUtil::Lower(mn)] = rel;
					}
				}
			}
			if (rel.empty() || scope.count(rel) == 0) {
				continue;
			}
			string ents = join_arr(yj::yyjson_obj_get(sm, "entities"), [&](yj::yyjson_val *e) {
				string nm = gstr(e, "name"), ty = gstr(e, "type"), ex = gstr(e, "expr");
				return (ex.empty() ? nm : ex) + (ty.empty() ? "" : " [" + ty + "]");
			});
			string dims = join_arr(yj::yyjson_obj_get(sm, "dimensions"), [&](yj::yyjson_val *e) {
				string nm = gstr(e, "name"), ty = gstr(e, "type"), ex = gstr(e, "expr");
				return (ex.empty() ? nm : ex) + (ty.empty() ? "" : " (" + ty + ")");
			});
			string meass = join_arr(meas, [&](yj::yyjson_val *e) {
				string nm = gstr(e, "name"), ag = gstr(e, "agg"), ex = gstr(e, "expr");
				return nm + " = " + ag + "(" + (ex.empty() ? nm : ex) + ")";
			});
			block += "- " + (smname.empty() ? rel : smname) + " (table " + rel + ")\n";
			if (!ents.empty()) {
				block += "    entities (join keys): " + ents + "\n";
			}
			if (!dims.empty()) {
				block += "    dimensions (group by): " + dims + "\n";
			}
			if (!meass.empty()) {
				block += "    measures (aggregate): " + meass + "\n";
			}
		}
		if (!block.empty()) {
			info.semantic_block =
			    "\nSemantic models (dbt) — JOIN on entities, GROUP BY dimensions, aggregate measures:\n" + block;
		}
	}

	// Metrics: MetricFlow (type + type_params) with a legacy fallback
	// (calculation_method + expression). Scoped to metrics whose measure belongs to an
	// in-scope semantic model when that's resolvable; otherwise included best-effort.
	yj::yyjson_val *metrics = yj::yyjson_obj_get(root, "metrics");
	if (metrics && yj::yyjson_is_obj(metrics)) {
		string block;
		yj::yyjson_obj_iter it;
		yj::yyjson_obj_iter_init(metrics, &it);
		yj::yyjson_val *key;
		while ((key = yj::yyjson_obj_iter_next(&it))) {
			yj::yyjson_val *m = yj::yyjson_obj_iter_get_val(key);
			string name = gstr(m, "name");
			if (name.empty()) {
				continue;
			}
			string type = gstr(m, "type");
			string defn;
			vector<string> ref_measures; // measures this metric is built on (for scoping)
			yj::yyjson_val *tp = yj::yyjson_obj_get(m, "type_params");
			if (tp) {
				yj::yyjson_val *measure = yj::yyjson_obj_get(tp, "measure");
				if (measure) {
					string mn =
					    yj::yyjson_is_str(measure) ? string(yj::yyjson_get_str(measure)) : gstr(measure, "name");
					if (!mn.empty()) {
						ref_measures.push_back(mn);
						defn = mn;
					}
				}
				if (defn.empty()) {
					string num = gstr(yj::yyjson_obj_get(tp, "numerator"), "name");
					string den = gstr(yj::yyjson_obj_get(tp, "denominator"), "name");
					if (!num.empty() && !den.empty()) {
						ref_measures.push_back(num);
						ref_measures.push_back(den);
						defn = num + " / " + den;
					}
				}
				if (defn.empty()) {
					defn = gstr(tp, "expr");
				}
			}
			if (defn.empty()) { // legacy dbt metrics
				string calc = gstr(m, "calculation_method");
				if (calc.empty()) {
					calc = type;
				}
				string expr = gstr(m, "expression");
				defn = (!calc.empty() && !expr.empty()) ? (calc + "(" + expr + ")") : expr;
			}
			// Scope: drop the metric if ANY measure it references belongs to a semantic model
			// outside the current scope (so a 'taxi'-scoped ASK never sees 'finance' metrics).
			// Metrics we can't resolve to a measure (e.g. derived) are kept best-effort.
			bool out_of_scope = false;
			for (auto &mn : ref_measures) {
				auto f = measure_rel.find(StringUtil::Lower(mn));
				if (f != measure_rel.end() && scope.count(f->second) == 0) {
					out_of_scope = true;
					break;
				}
			}
			if (out_of_scope) {
				continue;
			}
			string line = "- " + name;
			if (!type.empty()) {
				line += " [" + type + "]";
			}
			if (!defn.empty()) {
				line += " = " + defn;
			}
			string desc = gstr(m, "description");
			if (!desc.empty()) {
				line += "  -- " + desc;
			}
			block += line + "\n";
		}
		if (!block.empty()) {
			info.metrics_block = "\nMetrics (dbt) — use these definitions when the question names them:\n" + block;
		}
	}

	// Relationships (join keys) from `relationships` generic tests.
	yj::yyjson_val *nodes = yj::yyjson_obj_get(root, "nodes");
	if (nodes && yj::yyjson_is_obj(nodes)) {
		string block;
		yj::yyjson_obj_iter it;
		yj::yyjson_obj_iter_init(nodes, &it);
		yj::yyjson_val *key;
		while ((key = yj::yyjson_obj_iter_next(&it))) {
			yj::yyjson_val *node = yj::yyjson_obj_iter_get_val(key);
			if (gstr(node, "resource_type") != "test") {
				continue;
			}
			yj::yyjson_val *tm = yj::yyjson_obj_get(node, "test_metadata");
			if (!tm || gstr(tm, "name") != "relationships") {
				continue;
			}
			yj::yyjson_val *kw = yj::yyjson_obj_get(tm, "kwargs");
			string parent = StringUtil::Lower(ExtractRefName(gstr(kw, "to")));
			string field = gstr(kw, "field");
			string fk = gstr(kw, "column_name");
			// strip quotes/whitespace a dbt jinja rendering may leave around the FK name
			while (!fk.empty() && (fk.front() == '"' || fk.front() == ' ')) {
				fk.erase(fk.begin());
			}
			while (!fk.empty() && (fk.back() == '"' || fk.back() == ' ')) {
				fk.pop_back();
			}
			string child;
			yj::yyjson_val *dep = yj::yyjson_obj_get(node, "depends_on");
			yj::yyjson_val *dn = dep ? yj::yyjson_obj_get(dep, "nodes") : nullptr;
			if (dn && yj::yyjson_is_arr(dn)) {
				size_t n = yj::yyjson_arr_size(dn);
				for (size_t i = 0; i < n; i++) {
					yj::yyjson_val *e = yj::yyjson_arr_get(dn, i);
					if (e && yj::yyjson_is_str(e)) {
						auto f = id_relation.find(string(yj::yyjson_get_str(e)));
						if (f != id_relation.end() && f->second != parent) {
							child = f->second;
						}
					}
				}
			}
			if (parent.empty() || field.empty()) {
				continue;
			}
			if (scope.count(parent) == 0 && (child.empty() || scope.count(child) == 0)) {
				continue;
			}
			block += "- " + (child.empty() ? string("?") : child) + "." + (fk.empty() ? string("<fk>") : fk) + " -> " +
			         parent + "." + field + "\n";
		}
		if (!block.empty()) {
			info.relationships_block = "\nRelationships (join keys):\n" + block;
		}
	}

	info.loaded = true;
	return info; // doc stays cached — do not free
}

// Resolve a set of dbt tags to the relations (aliases, lowercased) that carry any of
// them — so a caller can scope ASK by DOMAIN ("taxi") instead of naming each table.
// Reads the cached manifest; matches both node-level `tags` and `config.tags`.
static vector<string> ResolveTagScope(const string &manifest_path, const std::set<string> &tags) {
	vector<string> out;
	if (tags.empty()) {
		return out;
	}
	yj::yyjson_doc *doc = GetCachedManifestDoc(manifest_path);
	if (!doc) {
		return out;
	}
	yj::yyjson_val *root = yj::yyjson_doc_get_root(doc);
	if (!root || !yj::yyjson_is_obj(root)) {
		return out;
	}
	auto get_str = [](yj::yyjson_val *o, const char *k) -> string {
		yj::yyjson_val *v = o ? yj::yyjson_obj_get(o, k) : nullptr;
		return (v && yj::yyjson_is_str(v)) ? string(yj::yyjson_get_str(v)) : string();
	};
	std::set<string> seen;
	for (const char *section : {"nodes", "sources"}) {
		yj::yyjson_val *sec = yj::yyjson_obj_get(root, section);
		if (!sec || !yj::yyjson_is_obj(sec)) {
			continue;
		}
		yj::yyjson_obj_iter it;
		yj::yyjson_obj_iter_init(sec, &it);
		yj::yyjson_val *key;
		while ((key = yj::yyjson_obj_iter_next(&it))) {
			yj::yyjson_val *node = yj::yyjson_obj_iter_get_val(key);
			bool hit = false;
			auto scan = [&](yj::yyjson_val *arr) {
				if (!arr || !yj::yyjson_is_arr(arr)) {
					return;
				}
				size_t n = yj::yyjson_arr_size(arr);
				for (size_t i = 0; i < n && !hit; i++) {
					yj::yyjson_val *e = yj::yyjson_arr_get(arr, i);
					if (e && yj::yyjson_is_str(e) && tags.count(StringUtil::Lower(string(yj::yyjson_get_str(e))))) {
						hit = true;
					}
				}
			};
			scan(yj::yyjson_obj_get(node, "tags"));
			yj::yyjson_val *cfg = yj::yyjson_obj_get(node, "config");
			if (cfg) {
				scan(yj::yyjson_obj_get(cfg, "tags"));
			}
			if (!hit) {
				continue;
			}
			string a = get_str(node, "alias");
			if (a.empty()) {
				a = get_str(node, "identifier");
			}
			if (a.empty()) {
				a = get_str(node, "name");
			}
			a = StringUtil::Lower(a);
			if (!a.empty() && seen.insert(a).second) {
				out.push_back(a);
			}
		}
	}
	return out;
}

// Every queryable relation (model/seed/snapshot/source) in the manifest, as lowercased
// aliases. Used as the candidate set when ASK is pointed at a manifest with no ON and no
// tags — the whole-warehouse case, where embedding retrieval does the narrowing.
static vector<string> AllManifestRelations(const string &manifest_path) {
	vector<string> out;
	yj::yyjson_doc *doc = GetCachedManifestDoc(manifest_path);
	yj::yyjson_val *root = doc ? yj::yyjson_doc_get_root(doc) : nullptr;
	if (!root || !yj::yyjson_is_obj(root)) {
		return out;
	}
	auto get_str = [](yj::yyjson_val *o, const char *k) -> string {
		yj::yyjson_val *v = o ? yj::yyjson_obj_get(o, k) : nullptr;
		return (v && yj::yyjson_is_str(v)) ? string(yj::yyjson_get_str(v)) : string();
	};
	std::set<string> seen;
	for (const char *section : {"nodes", "sources"}) {
		yj::yyjson_val *sec = yj::yyjson_obj_get(root, section);
		if (!sec || !yj::yyjson_is_obj(sec)) {
			continue;
		}
		yj::yyjson_obj_iter it;
		yj::yyjson_obj_iter_init(sec, &it);
		yj::yyjson_val *key;
		while ((key = yj::yyjson_obj_iter_next(&it))) {
			yj::yyjson_val *node = yj::yyjson_obj_iter_get_val(key);
			string rt = get_str(node, "resource_type");
			if (rt != "model" && rt != "seed" && rt != "snapshot" && rt != "source") {
				continue;
			}
			string a = get_str(node, "alias");
			if (a.empty()) {
				a = get_str(node, "identifier");
			}
			if (a.empty()) {
				a = get_str(node, "name");
			}
			a = StringUtil::Lower(a);
			if (!a.empty() && seen.insert(a).second) {
				out.push_back(a);
			}
		}
	}
	return out;
}

// Candidate list for the router: each in-scope relation with its dbt description (if any).
static vector<std::pair<string, string>> CandidateCatalog(ClientContext &context, const string &manifest_path,
                                                          const vector<string> &aliases) {
	// Rich per-table summary for retrieval: the table's description PLUS its semantic-layer
	// measures and the metrics built on them. Bare table names embed/route poorly; the
	// semantic layer is exactly the signal that tells "gross margin" from "inventory".
	std::map<string, string> desc, meas, mets;
	if (!manifest_path.empty()) {
		yj::yyjson_doc *doc = GetCachedManifestDoc(manifest_path);
		yj::yyjson_val *root = doc ? yj::yyjson_doc_get_root(doc) : nullptr;
		if (root && yj::yyjson_is_obj(root)) {
			std::set<string> want;
			for (auto &a : aliases) {
				want.insert(StringUtil::Lower(a));
			}
			auto get_str = [](yj::yyjson_val *o, const char *k) -> string {
				yj::yyjson_val *v = o ? yj::yyjson_obj_get(o, k) : nullptr;
				return (v && yj::yyjson_is_str(v)) ? string(yj::yyjson_get_str(v)) : string();
			};
			auto join_names = [&](yj::yyjson_val *arr) {
				string s;
				if (arr && yj::yyjson_is_arr(arr)) {
					size_t n = yj::yyjson_arr_size(arr);
					for (size_t i = 0; i < n; i++) {
						string nm = get_str(yj::yyjson_arr_get(arr, i), "name");
						if (!nm.empty()) {
							s += (s.empty() ? "" : ", ") + nm;
						}
					}
				}
				return s;
			};
			for (const char *section : {"nodes", "sources"}) {
				yj::yyjson_val *sec = yj::yyjson_obj_get(root, section);
				if (!sec || !yj::yyjson_is_obj(sec)) {
					continue;
				}
				yj::yyjson_obj_iter it;
				yj::yyjson_obj_iter_init(sec, &it);
				yj::yyjson_val *key;
				while ((key = yj::yyjson_obj_iter_next(&it))) {
					yj::yyjson_val *node = yj::yyjson_obj_iter_get_val(key);
					string a = get_str(node, "alias");
					if (a.empty()) {
						a = get_str(node, "identifier");
					}
					if (a.empty()) {
						a = get_str(node, "name");
					}
					a = StringUtil::Lower(a);
					string d = get_str(node, "description");
					if (!a.empty() && !d.empty() && want.count(a)) {
						desc[a] = d;
					}
				}
			}
			// Semantic models: description (fallback) + measure names; also measure -> alias.
			std::map<string, string> measure_alias;
			yj::yyjson_val *sms = yj::yyjson_obj_get(root, "semantic_models");
			if (sms && yj::yyjson_is_obj(sms)) {
				yj::yyjson_obj_iter it;
				yj::yyjson_obj_iter_init(sms, &it);
				yj::yyjson_val *key;
				while ((key = yj::yyjson_obj_iter_next(&it))) {
					yj::yyjson_val *sm = yj::yyjson_obj_iter_get_val(key);
					yj::yyjson_val *nr = yj::yyjson_obj_get(sm, "node_relation");
					string a = nr ? StringUtil::Lower(get_str(nr, "alias")) : string();
					yj::yyjson_val *measures = yj::yyjson_obj_get(sm, "measures");
					if (measures && yj::yyjson_is_arr(measures) && !a.empty()) {
						size_t n = yj::yyjson_arr_size(measures);
						for (size_t i = 0; i < n; i++) {
							string mn = get_str(yj::yyjson_arr_get(measures, i), "name");
							if (!mn.empty()) {
								measure_alias[StringUtil::Lower(mn)] = a;
							}
						}
					}
					if (a.empty() || !want.count(a)) {
						continue;
					}
					if (desc[a].empty()) {
						desc[a] = get_str(sm, "description");
					}
					meas[a] = join_names(measures);
				}
			}
			// Attach each simple metric's label to the table its measure lives on.
			yj::yyjson_val *metrics = yj::yyjson_obj_get(root, "metrics");
			if (metrics && yj::yyjson_is_obj(metrics)) {
				yj::yyjson_obj_iter it;
				yj::yyjson_obj_iter_init(metrics, &it);
				yj::yyjson_val *key;
				while ((key = yj::yyjson_obj_iter_next(&it))) {
					yj::yyjson_val *m = yj::yyjson_obj_iter_get_val(key);
					if (get_str(m, "type") != "simple") {
						continue;
					}
					yj::yyjson_val *tp = yj::yyjson_obj_get(m, "type_params");
					string mn = tp ? get_str(yj::yyjson_obj_get(tp, "measure"), "name") : string();
					auto f = measure_alias.find(StringUtil::Lower(mn));
					if (f == measure_alias.end() || !want.count(f->second)) {
						continue;
					}
					string lbl = get_str(m, "label");
					if (lbl.empty()) {
						lbl = get_str(m, "name");
					}
					if (!lbl.empty()) {
						mets[f->second] += (mets[f->second].empty() ? "" : ", ") + lbl;
					}
				}
			}
		}
	}
	// For candidates the semantic layer didn't enrich (plain DuckDB with no dbt, staging
	// tables, etc.), fall back to the table's COLUMN names from the catalog so retrieval
	// still has real signal. One grouped query for all such tables.
	std::map<string, string> cols;
	{
		string inlist;
		for (auto &a : aliases) {
			string al = StringUtil::Lower(a);
			if (meas[al].empty()) {
				inlist += (inlist.empty() ? "" : ", ") + SqlLit(al);
			}
		}
		if (!inlist.empty()) {
			Connection con(DatabaseInstance::GetDatabase(context));
			auto res = con.Query("SELECT lower(table_name), string_agg(column_name, ', ' ORDER BY ordinal_position) "
			                     "FROM information_schema.columns WHERE lower(table_name) IN (" +
			                     inlist + ") GROUP BY 1");
			if (res && !res->HasError()) {
				for (idx_t r = 0; r < res->RowCount(); r++) {
					string cs = res->GetValue(1, r).ToString();
					if (cs.size() > 300) {
						cs = cs.substr(0, 300) + "…";
					}
					cols[res->GetValue(0, r).ToString()] = cs;
				}
			}
		}
	}

	vector<std::pair<string, string>> out;
	for (auto &a : aliases) {
		string al = StringUtil::Lower(a);
		string s = desc.count(al) ? desc[al] : string();
		if (!meas[al].empty()) {
			s += (s.empty() ? "" : " — ") + ("measures: " + meas[al]);
		} else if (cols.count(al)) {
			s += (s.empty() ? "" : " — ") + ("columns: " + cols[al]);
		}
		if (!mets[al].empty()) {
			s += "; metrics: " + mets[al];
		}
		out.emplace_back(a, s);
	}
	return out;
}

// Two-stage retrieval: when a scope resolves to many tables (e.g. a whole tagged domain),
// ask the model FIRST which tables this specific question needs, and keep only those, so
// the schema we send stays small and focused. One cheap routing call (memoized). Returns
// the chosen aliases; empty on any failure so the caller safely keeps the full set.
static vector<string> RouteTables(const duckthink::LlmCall &call, const string &request,
                                  const vector<std::pair<string, string>> &candidates) {
	string list;
	for (auto &c : candidates) {
		list += "- " + c.first + (c.second.empty() ? "" : (": " + c.second)) + "\n";
	}
	string sys = "You select which database tables are needed to answer a question. Output ONLY a JSON "
	             "array of table names — a subset of the candidates, minimal but sufficient. No prose, no fences.";
	string user = "Question: " + request + "\n\nCandidate tables:\n" + list;
	string resp = StripSqlFences(ExtractChat(call, sys, user));

	std::set<string> valid;
	for (auto &c : candidates) {
		valid.insert(StringUtil::Lower(c.first));
	}
	vector<string> picked;
	yj::yyjson_doc *doc = yj::yyjson_read(resp.c_str(), resp.size(), 0);
	if (doc) {
		yj::yyjson_val *root = yj::yyjson_doc_get_root(doc);
		if (root && yj::yyjson_is_arr(root)) {
			std::set<string> seen;
			size_t n = yj::yyjson_arr_size(root);
			for (size_t i = 0; i < n; i++) {
				yj::yyjson_val *e = yj::yyjson_arr_get(root, i);
				if (e && yj::yyjson_is_str(e)) {
					string nm = StringUtil::Lower(string(yj::yyjson_get_str(e)));
					if (valid.count(nm) && seen.insert(nm).second) {
						picked.push_back(nm);
					}
				}
			}
		}
		yj::yyjson_doc_free(doc);
	}
	return picked;
}

static double CosineSim(const std::vector<float> &a, const std::vector<float> &b) {
	if (a.empty() || b.empty() || a.size() != b.size()) {
		return -1.0;
	}
	double dot = 0, na = 0, nb = 0;
	for (size_t i = 0; i < a.size(); i++) {
		dot += (double)a[i] * b[i];
		na += (double)a[i] * a[i];
		nb += (double)b[i] * b[i];
	}
	if (na <= 0 || nb <= 0) {
		return -1.0;
	}
	return dot / (std::sqrt(na) * std::sqrt(nb));
}

// Table-summary embeddings, cached process-wide by (manifest path, mtime, embed model) so a
// big domain's tables are embedded ONCE, not on every question.
static std::mutex g_embcache_mu;
static std::map<string, std::map<string, std::vector<float>>> g_embcache;

// Stage-1 vector recall: rank the candidate tables by cosine of their summary to the
// question and keep the top k. Summaries are embedded once (cached); only the question is
// embedded per call. Scales to very large domains. Returns aliases; empty on failure.
static vector<string> EmbedRetrieve(const duckthink::LlmCall &embed_call, const string &manifest_path,
                                    const string &request, const vector<std::pair<string, string>> &candidates,
                                    idx_t k) {
	if (candidates.empty()) {
		return {};
	}
	int64_t mtime = 0;
	struct stat st;
	if (stat(manifest_path.c_str(), &st) == 0) {
		mtime = (int64_t)st.st_mtime;
	}
	string key = manifest_path + "\x1f" + std::to_string(mtime) + "\x1f" + embed_call.model_id;

	// Embed any candidate summaries not yet cached.
	vector<string> miss_alias, miss_text;
	{
		std::lock_guard<std::mutex> lk(g_embcache_mu);
		auto &cache = g_embcache[key];
		for (auto &c : candidates) {
			string a = StringUtil::Lower(c.first);
			if (cache.find(a) == cache.end()) {
				miss_alias.push_back(a);
				miss_text.push_back(c.second.empty() ? c.first : (c.first + " — " + c.second));
			}
		}
	}
	if (!miss_text.empty()) {
		auto vecs = duckthink::LlmEmbed(embed_call, miss_text);
		std::lock_guard<std::mutex> lk(g_embcache_mu);
		auto &cache = g_embcache[key];
		for (size_t i = 0; i < miss_alias.size() && i < vecs.size(); i++) {
			cache[miss_alias[i]] = std::move(vecs[i]);
		}
	}

	auto qv = duckthink::LlmEmbed(embed_call, {request});
	if (qv.empty() || qv[0].empty()) {
		return {};
	}

	vector<std::pair<double, string>> scored;
	{
		std::lock_guard<std::mutex> lk(g_embcache_mu);
		auto &cache = g_embcache[key];
		for (auto &c : candidates) {
			auto it = cache.find(StringUtil::Lower(c.first));
			if (it != cache.end()) {
				scored.emplace_back(CosineSim(qv[0], it->second), StringUtil::Lower(c.first));
			}
		}
	}
	std::sort(scored.begin(), scored.end(),
	          [](const std::pair<double, string> &x, const std::pair<double, string> &y) { return x.first > y.first; });
	vector<string> top;
	for (size_t i = 0; i < scored.size() && i < k; i++) {
		top.push_back(scored[i].second);
	}
	return top;
}

// Introspect the named tables and ask the model for one SQL query. `repair`, when
// non-empty, is appended to the prompt to feed a previous failure back to the
// model (self-correction). Returns the raw SQL (validation happens in the caller).
static string GenerateAskSql(ClientContext &context, const string &request, const string &tables_csv,
                             const string &return_schema, const string &repair) {
	Value mv;
	if (!(context.TryGetCurrentSetting(kLlmModelSetting, mv) && !mv.IsNull() && !mv.ToString().empty())) {
		throw BinderException("ASK: no model configured. SET ask_model='<name>' (declared via CREATE MODEL).");
	}
	auto call = ResolveModelCall(context, mv.ToString());

	// Collect the in-scope table names, and (if a dbt manifest is configured) the
	// semantic context — descriptions, synonyms, metrics, join keys — for them.
	vector<string> table_list;
	std::set<string> scope;
	for (auto &tp : SplitTopLevel(tables_csv, ',')) {
		string t = TrimStr(tp);
		if (!t.empty()) {
			table_list.push_back(t);
			scope.insert(StringUtil::Lower(t));
		}
	}

	Value dm;
	bool have_manifest =
	    context.TryGetCurrentSetting("ask_dbt_semantic_manifest", dm) && !dm.IsNull() && !dm.ToString().empty();

	// Domain scoping: SET ask_dbt_scope_tags='taxi' pulls in every model/source carrying that
	// dbt tag, so a service names its DOMAIN once instead of listing tables in ON (...).
	// Tag-resolved relations are unioned with any explicit ON tables.
	Value stv;
	if (have_manifest && context.TryGetCurrentSetting("ask_dbt_scope_tags", stv) && !stv.IsNull() &&
	    !stv.ToString().empty()) {
		std::set<string> tags;
		for (auto &tp : SplitTopLevel(stv.ToString(), ',')) {
			string t = TrimStr(tp);
			if (!t.empty()) {
				tags.insert(StringUtil::Lower(t));
			}
		}
		for (auto &alias : ResolveTagScope(dm.ToString(), tags)) {
			if (scope.insert(alias).second) {
				table_list.push_back(alias);
			}
		}
	}

	// Whole-warehouse mode: no ON and no tags. Every relation becomes a candidate and the
	// two-stage retrieval below narrows to the relevant few — "just ask" over a big schema.
	// With a dbt manifest the candidates come from it (rich summaries); with no dbt at all we
	// fall back to the plain DuckDB catalog (candidates enriched with their column names).
	if (table_list.empty() && have_manifest) {
		for (auto &a : AllManifestRelations(dm.ToString())) {
			if (scope.insert(a).second) {
				table_list.push_back(a);
			}
		}
	} else if (table_list.empty()) {
		Connection con(DatabaseInstance::GetDatabase(context));
		auto res = con.Query("SELECT table_name FROM information_schema.tables "
		                     "WHERE table_schema NOT IN ('information_schema', 'pg_catalog', 'system') "
		                     "ORDER BY table_name");
		if (res && !res->HasError()) {
			for (idx_t r = 0; r < res->RowCount(); r++) {
				string t = res->GetValue(0, r).ToString();
				if (scope.insert(StringUtil::Lower(t)).second) {
					table_list.push_back(t);
				}
			}
		}
	}

	if (table_list.empty()) {
		throw BinderException("ASK: no tables in scope — the database has no tables, or list them in ON (...) / "
		                      "SET ask_dbt_scope_tags / SET ask_dbt_semantic_manifest.");
	}

	// Two-stage retrieval: when the scope is large (e.g. a whole tagged domain resolved
	// without an ON clause), let the model pick the relevant tables first so the schema we
	// send stays small and focused. Threshold configurable; 0 disables the step.
	Value mtv;
	int64_t max_tables = 12;
	if (context.TryGetCurrentSetting("ask_scope_max_tables", mtv) && !mtv.IsNull()) {
		max_tables = mtv.GetValue<int64_t>();
	}
	if (max_tables > 0 && (int64_t)table_list.size() > max_tables) {
		string manifest = have_manifest ? dm.ToString() : string();

		// Stage 1 — vector recall. If an embedding model is configured, rank ALL candidates
		// by cosine of their (cached) summary to the question and keep the top ask_retrieve_k.
		// This is the scalable coarse filter: embedding is cheap per candidate, summaries are
		// embedded once, so a 1000-table domain narrows to a few dozen without a giant prompt.
		Value emv;
		if (context.TryGetCurrentSetting("ask_embed_model", emv) && !emv.IsNull() && !emv.ToString().empty()) {
			int64_t retrieve_k = 30;
			Value rkv;
			if (context.TryGetCurrentSetting("ask_retrieve_k", rkv) && !rkv.IsNull()) {
				retrieve_k = rkv.GetValue<int64_t>();
			}
			if (retrieve_k > 0 && (int64_t)table_list.size() > retrieve_k) {
				auto embed_call = ResolveModelCall(context, emv.ToString());
				auto recalled = EmbedRetrieve(embed_call, manifest, request,
				                              CandidateCatalog(context, manifest, table_list), (idx_t)retrieve_k);
				if (!recalled.empty()) {
					table_list = recalled;
				}
			}
		}

		// Stage 2 — LLM rerank. Pick the minimal relevant set from the survivors (names +
		// descriptions), capped at max_tables so the schema we send stays small.
		if ((int64_t)table_list.size() > max_tables) {
			auto picked = RouteTables(call, request, CandidateCatalog(context, manifest, table_list));
			if (!picked.empty()) {
				if ((int64_t)picked.size() > max_tables) {
					picked.resize((size_t)max_tables);
				}
				table_list = picked;
			}
		}

		scope.clear();
		for (auto &t : table_list) {
			scope.insert(StringUtil::Lower(t));
		}
	}

	DbtInfo dbt;
	if (have_manifest) {
		dbt = LoadDbtInfo(dm.ToString(), scope);
	}

	Connection con(DatabaseInstance::GetDatabase(context));
	string schema_block;
	for (auto &t : table_list) {
		auto res = con.Query("SELECT column_name, data_type FROM information_schema.columns "
		                     "WHERE lower(table_name) = lower(" +
		                     SqlLit(t) + ") ORDER BY ordinal_position");
		if (res->HasError()) {
			throw BinderException("ASK: could not read schema of '%s': %s", t, res->GetError());
		}
		if (res->RowCount() == 0) {
			throw BinderException("ASK: table '%s' not found (no columns in information_schema)", t);
		}
		string tl = StringUtil::Lower(t);
		auto ci = dbt.loaded ? dbt.col_desc.find(tl) : dbt.col_desc.end();
		bool enrich = ci != dbt.col_desc.end();
		if (!enrich) {
			string cols;
			for (idx_t r = 0; r < res->RowCount(); r++) {
				cols += (r ? ", " : "") + res->GetValue(0, r).ToString() + " " + res->GetValue(1, r).ToString();
			}
			schema_block += "CREATE TABLE " + t + "(" + cols + ");\n";
		} else {
			// Multi-line so each column can carry its dbt description/synonyms.
			schema_block += "CREATE TABLE " + t + "(\n";
			for (idx_t r = 0; r < res->RowCount(); r++) {
				string cname = res->GetValue(0, r).ToString();
				schema_block += "  " + cname + " " + res->GetValue(1, r).ToString();
				if (r + 1 < res->RowCount()) {
					schema_block += ",";
				}
				auto d = ci->second.find(StringUtil::Lower(cname));
				if (d != ci->second.end()) {
					schema_block += "  -- " + d->second;
				}
				schema_block += "\n";
			}
			schema_block += ");\n";
		}
		if (dbt.loaded) {
			auto td = dbt.table_desc.find(tl);
			if (td != dbt.table_desc.end()) {
				schema_block += "-- " + t + ": " + td->second + "\n";
			}
		}
	}
	if (dbt.loaded) {
		schema_block += dbt.semantic_block + dbt.metrics_block + dbt.relationships_block;
	}

	string sys = "You are an expert DuckDB SQL engineer. Given the schema, write EXACTLY ONE DuckDB SQL SELECT "
	             "query that answers the request. Output ONLY the SQL — no prose, no comments, no markdown fences, "
	             "no trailing semicolon. Use only the given tables and columns. The result columns, in order, MUST "
	             "be exactly: " +
	             return_schema + ".";
	string user = "Schema:\n" + schema_block + "\nRequest: " + request;
	if (!repair.empty()) {
		user += "\n\n" + repair;
	}
	string sql = StripSqlFences(ExtractChat(call, sys, user));

	Value lg;
	if (context.TryGetCurrentSetting("duckthink_log", lg) && !lg.IsNull() && lg.GetValue<bool>()) {
		fprintf(stderr, "[duckthink] ASK SQL: %s\n", sql.c_str());
	}
	return sql;
}

// ASK holds the request params in bind (cheap, no LLM); the generation + execution
// happen once in init, and the generated query's result is STREAMED in the scan
// (chunk by chunk) rather than materialized — constant memory, like a normal query.
struct AskBindData : public TableFunctionData {
	string request, tables, schema;
	vector<LogicalType> types; // = RETURN types (also set on the function)
};
struct AskGlobalState : public GlobalTableFunctionState {
	unique_ptr<Connection> con;     // kept alive while the stream is consumed
	unique_ptr<QueryResult> result; // streaming result of the generated query
	unique_ptr<DataChunk> pending;  // first chunk, fetched in init to surface early errors
	bool finished = false;
};

static unique_ptr<FunctionData> AskBind(ClientContext &context, TableFunctionBindInput &input,
                                        vector<LogicalType> &return_types, vector<string> &names) {
	auto bind = make_uniq<AskBindData>();
	bind->request = StringValue::Get(input.inputs[0]);
	bind->tables = StringValue::Get(input.inputs[1]);
	bind->schema = StringValue::Get(input.inputs[2]);
	ParseReturnSchema(context, bind->schema, names, return_types);
	bind->types = return_types;
	return std::move(bind);
}

// Generate + self-correct + start streaming, once. On a bad generation (non-SELECT,
// execution/bind error, wrong column count) the error is fed back to the model and
// it retries, up to ask_max_retries times.
static unique_ptr<GlobalTableFunctionState> AskInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind = input.bind_data->Cast<AskBindData>();
	auto g = make_uniq<AskGlobalState>();

	int max_retries = 2;
	Value rv;
	if (context.TryGetCurrentSetting("ask_max_retries", rv) && !rv.IsNull()) {
		max_retries = MaxValue<int>(0, (int)rv.GetValue<int64_t>());
	}

	auto con = make_uniq<Connection>(DatabaseInstance::GetDatabase(context));
	string repair, last_err, sql;
	for (int attempt = 0;; attempt++) {
		sql = GenerateAskSql(context, bind.request, bind.tables, bind.schema, repair);
		string cleaned, why;
		if (!IsReadOnlySelect(sql, cleaned, why)) {
			last_err = "generated query " + why;
			repair = "Your previous answer was rejected because " + why +
			         ". Return exactly ONE read-only SELECT/WITH query and nothing else.\nRejected SQL:\n" + sql;
		} else {
			auto res = con->SendQuery(cleaned); // streaming; bind errors surface here
			if (res->HasError()) {
				last_err = res->GetError();
				repair = "Your previous SQL failed. Fix it and return only the corrected SQL.\nSQL:\n" + cleaned +
				         "\nError: " + last_err;
			} else if (res->types.size() != bind.types.size()) {
				last_err = "returned " + std::to_string(res->types.size()) + " columns but " +
				           std::to_string(bind.types.size()) + " were requested";
				repair = "Your previous query " + last_err +
				         ". Return exactly these columns, in order: " + bind.schema + ".\nSQL:\n" + cleaned;
			} else {
				auto first = res->Fetch(); // begin execution; catch immediate runtime errors
				if (res->HasError()) {
					last_err = res->GetError();
					repair = "Your previous SQL failed while running. Fix it and return only the corrected "
					         "SQL.\nSQL:\n" +
					         cleaned + "\nError: " + last_err;
				} else {
					g->con = std::move(con);
					g->result = std::move(res);
					g->pending = std::move(first);
					return std::move(g);
				}
			}
		}
		if (attempt >= max_retries) {
			throw BinderException("ASK: gave up after %d attempt(s). Last error: %s\nLast SQL:\n%s", attempt + 1,
			                      last_err, sql);
		}
	}
}

static void AskScan(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &g = data.global_state->Cast<AskGlobalState>();
	auto &bind = data.bind_data->Cast<AskBindData>();
	if (g.finished) {
		output.SetCardinality(0);
		return;
	}
	unique_ptr<DataChunk> chunk;
	if (g.pending) {
		chunk = std::move(g.pending); // the first chunk fetched in init
	} else {
		chunk = g.result->Fetch();
		if (g.result->HasError()) {
			throw InvalidInputException("ASK: the generated query failed during execution: %s", g.result->GetError());
		}
	}
	if (!chunk || chunk->size() == 0) {
		g.finished = true;
		output.SetCardinality(0);
		return;
	}
	idx_t n = chunk->size();
	for (idx_t c = 0; c < bind.types.size(); c++) {
		for (idx_t r = 0; r < n; r++) {
			output.SetValue(c, r, chunk->GetValue(c, r).DefaultCastAs(bind.types[c]));
		}
	}
	output.SetCardinality(n);
}

// ask_sql: the dry run — one row, the raw generated SQL, without running it (so you
// can inspect what ASK would execute, including anything it would refuse).
struct AskSqlBindData : public TableFunctionData {
	string sql;
};
struct OneRowState : public GlobalTableFunctionState {
	bool done = false;
};
static unique_ptr<GlobalTableFunctionState> OneRowInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<OneRowState>();
}
static unique_ptr<FunctionData> AskSqlBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	string request = StringValue::Get(input.inputs[0]);
	string tables = StringValue::Get(input.inputs[1]);
	string schema = StringValue::Get(input.inputs[2]);
	names = {"sql"};
	return_types = {LogicalType::VARCHAR};
	auto bind = make_uniq<AskSqlBindData>();
	bind->sql = GenerateAskSql(context, request, tables, schema, "");
	return std::move(bind);
}
static void AskSqlScan(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &g = data.global_state->Cast<OneRowState>();
	if (g.done) {
		output.SetCardinality(0);
		return;
	}
	output.SetValue(0, 0, Value(data.bind_data->Cast<AskSqlBindData>().sql));
	output.SetCardinality(1);
	g.done = true;
}

} // namespace

void RegisterAiFunctions(ExtensionLoader &loader) {
	// LLM JOIN knobs (all overridable per-call via named parameters on llm_join).
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());
	config.AddExtensionOption(kLlmModelSetting, "Registered model ASK uses to generate SQL (see CREATE MODEL)",
	                          LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption(kLlmRateLimitWaitSetting,
	                          "Seconds to keep waiting out HTTP 429 rate limits (honoring the server's "
	                          "Retry-After) before failing a call. 0 = fail on the first 429.",
	                          LogicalType::BIGINT, Value::BIGINT(300));
	config.AddExtensionOption("llm_max_tokens",
	                          "Max response tokens per LLM call (required by Anthropic; ignored by OpenAI/Ollama).",
	                          LogicalType::BIGINT, Value::BIGINT(2048));

	for (auto *provider : kLlmProviders) {
		RegisterLlmSecretType(loader, provider);
	}

	// Model registry: CREATE [OR REPLACE] MODEL(...) [IF NOT EXISTS] / DROP MODEL — ASK needs
	// a declared model + a provider secret to talk to.
	ScalarFunctionSet create_model("create_model");
	create_model.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                        LogicalType::VARCHAR, CreateModelFun));
	create_model.AddFunction(
	    ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BOOLEAN},
	                   LogicalType::VARCHAR, CreateModelFun));
	loader.RegisterFunction(create_model);
	loader.RegisterFunction(ScalarFunction("drop_model", {LogicalType::VARCHAR}, LogicalType::VARCHAR, DropModelFun));

	// llm_models() — list the declared models.
	TableFunction models("llm_models", {}, LlmModelsScan, LlmModelsBind, LlmModelsInit);
	loader.RegisterFunction(models);

	// duckthink_metrics() — per-model API call / token / cache-hit counters for ASK.
	TableFunction metrics("duckthink_metrics", {}, MetricsScan, MetricsBind, LlmModelsInit);
	loader.RegisterFunction(metrics);
	loader.RegisterFunction(ScalarFunction("duckthink_metrics_reset", {}, LogicalType::VARCHAR, MetricsResetFun));

	// ASK / ask_sql — natural language -> SQL (the parser rewrites the keyword form).
	TableFunction ask("ask", {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, AskScan, AskBind,
	                  AskInit);
	loader.RegisterFunction(ask);
	TableFunction ask_sql("ask_sql", {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, AskSqlScan,
	                      AskSqlBind, OneRowInit);
	loader.RegisterFunction(ask_sql);
}

} // namespace duckdb

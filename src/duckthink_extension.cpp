#define DUCKDB_EXTENSION_MAIN

#include "duckthink_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/parser_extension.hpp"

#include "duckthink_internal.hpp"

#include <cctype>
#include <string>
#include <vector>

namespace duckdb {

//===--------------------------------------------------------------------===//
// Keyword rewriting for ASK and the model DDL
//
// duckthink is an ASK-only extension: `ASK('question') ON (t...) RETURN (...)`
// natural-language-to-SQL, plus the minimal object DDL it needs (CREATE/DROP
// MODEL). Both are surfaced as keyword syntax and lowered to the backing table
// / scalar functions here, in a parser_override that only runs after DuckDB's
// own parser has already failed (allow_parser_override_extension=FALLBACK), so
// it never shadows valid SQL.
//===--------------------------------------------------------------------===//

// Wrap a string as a single-quoted SQL literal (doubling embedded quotes).
static std::string SqlQuote(const std::string &s) {
	std::string out = "'";
	for (char c : s) {
		if (c == '\'') {
			out += "''";
		} else {
			out += c;
		}
	}
	out += "'";
	return out;
}

// A minimal lexer token; keeps byte offsets so we can splice the query in place.
struct SjTok {
	size_t start;
	size_t end;
	std::string lword; // lowercased text (words only)
	bool is_word;
	bool is_string; // single-quoted string literal
};

// Tokenize enough SQL to find our keywords OUTSIDE string literals and comments
// (so "ask" inside a string never triggers a rewrite). Words include dotted
// identifiers (a.txt) and tolerate double-quoted segments ("db"."tbl").
static vector<SjTok> SjLex(const std::string &s) {
	vector<SjTok> toks;
	size_t i = 0, n = s.size();
	while (i < n) {
		char c = s[i];
		if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') {
			i++;
			continue;
		}
		if (c == '-' && i + 1 < n && s[i + 1] == '-') { // line comment
			while (i < n && s[i] != '\n') {
				i++;
			}
			continue;
		}
		if (c == '/' && i + 1 < n && s[i + 1] == '*') { // block comment
			i += 2;
			while (i + 1 < n && !(s[i] == '*' && s[i + 1] == '/')) {
				i++;
			}
			i = (i + 1 < n) ? i + 2 : n;
			continue;
		}
		if (c == '\'') { // string literal (single quotes)
			size_t st = i++;
			while (i < n) {
				if (s[i] == '\'') {
					if (i + 1 < n && s[i + 1] == '\'') { // escaped quote
						i += 2;
						continue;
					}
					i++;
					break;
				}
				i++;
			}
			toks.push_back({st, i, std::string(), false, true});
			continue;
		}
		// Word / dotted identifier, tolerating double-quoted segments so a qualified
		// name like "db"."schema"."table" is one word token, not three.
		if (std::isalnum((unsigned char)c) || c == '_' || c == '.' || c == '$' || c == '"') {
			size_t st = i;
			while (i < n) {
				char d = s[i];
				if (d == '"') { // quoted segment
					i++;
					while (i < n) {
						if (s[i] == '"') {
							if (i + 1 < n && s[i + 1] == '"') { // escaped quote inside identifier
								i += 2;
								continue;
							}
							i++;
							break;
						}
						i++;
					}
				} else if (std::isalnum((unsigned char)d) || d == '_' || d == '.' || d == '$') {
					i++;
				} else {
					break;
				}
			}
			toks.push_back({st, i, StringUtil::Lower(s.substr(st, i - st)), true, false});
			continue;
		}
		toks.push_back({i, i + 1, std::string(), false, false}); // any other single char
		i++;
	}
	return toks;
}

// Strip surrounding quotes and unescape doubled quotes from a string token.
static std::string SjUnquote(const std::string &raw) {
	if (raw.size() < 2) {
		return raw;
	}
	char q = raw[0];
	std::string out;
	size_t last = raw.size() - 1;
	for (size_t i = 1; i < last;) {
		if (raw[i] == q && i + 1 < last && raw[i + 1] == q) {
			out += q;
			i += 2;
		} else {
			out += raw[i];
			i++;
		}
	}
	return out;
}

// Reparse the rewritten query and hand DuckDB the resulting statements.
static ParserOverrideResult ReparseAs(const std::string &rewritten, ParserOptions &options) {
	try {
		Parser parser(options);
		parser.ParseQuery(rewritten);
		return ParserOverrideResult(std::move(parser.statements));
	} catch (std::exception &e) {
		return ParserOverrideResult(e);
	}
}

static ParserOverrideResult DuckthinkParserOverride(ParserExtensionInfo *info, const string &query,
                                                    ParserOptions &options) {
	// Runs only after DuckDB's own parser failed (FALLBACK), so our fragments land
	// here without shadowing valid SQL. The query may be a multi-statement batch,
	// so we rewrite our fragments IN PLACE. `rewrites` counts any rewrite we make.
	std::string cur = query;
	std::string error;
	int rewrites = 0;

	// Object DDL -> its backing function, in place (multi-statement batches ok):
	//   CREATE [OR REPLACE] MODEL (...) [IF NOT EXISTS] -> SELECT create_model(...)
	//   DROP MODEL [IF EXISTS] <name>                   -> SELECT drop_model('<name>')
	for (int guard = 0; guard < 64; guard++) {
		auto toks = SjLex(cur);
		auto word = [&](size_t i) {
			return i < toks.size() && toks[i].is_word;
		};
		auto lw = [&](size_t i) {
			return word(i) ? toks[i].lword : std::string();
		};
		auto text = [&](size_t i) {
			return cur.substr(toks[i].start, toks[i].end - toks[i].start);
		};
		auto punct = [&](size_t i, const char *p) {
			return i < toks.size() && !toks[i].is_word && !toks[i].is_string &&
			       cur.compare(toks[i].start, toks[i].end - toks[i].start, p) == 0;
		};
		auto is_obj = [](const std::string &w) {
			return w == "model";
		};

		int found = -1;
		bool is_drop = false;
		bool or_replace = false;
		std::string obj;
		size_t kw = 0; // index of the MODEL keyword token
		for (size_t j = 0; j + 1 < toks.size(); j++) {
			if (lw(j) == "create" && lw(j + 1) == "or" && lw(j + 2) == "replace" && is_obj(lw(j + 3))) {
				found = (int)j;
				kw = j + 3;
				or_replace = true;
				obj = lw(j + 3);
				break;
			}
			if (lw(j) == "create" && is_obj(lw(j + 1))) {
				found = (int)j;
				kw = j + 1;
				obj = lw(j + 1);
				break;
			}
			if (lw(j) == "drop" && is_obj(lw(j + 1))) {
				found = (int)j;
				kw = j + 1;
				is_drop = true;
				obj = lw(j + 1);
				break;
			}
		}
		if (found < 0) {
			break;
		}

		if (is_drop) {
			size_t n = kw + 1;
			if (lw(n) == "if" && lw(n + 1) == "exists") {
				n += 2;
			}
			if (n >= toks.size()) {
				error = "DROP " + obj + ": expected a name";
				break;
			}
			std::string name = toks[n].is_string ? SjUnquote(text(n)) : text(n);
			cur = cur.substr(0, toks[found].start) + "SELECT drop_" + obj + "(" + SqlQuote(name) + ")" +
			      cur.substr(toks[n].end);
			rewrites++;
			continue;
		}

		// CREATE: an optional `IF NOT EXISTS` may sit between the keyword and the `(`.
		size_t p = kw + 1;
		bool if_not_exists = false;
		if (lw(p) == "if" && lw(p + 1) == "not" && lw(p + 2) == "exists") {
			if_not_exists = true;
			p += 3;
		}
		if (or_replace && if_not_exists) {
			error = "CREATE " + obj + ": cannot combine OR REPLACE with IF NOT EXISTS";
			break;
		}
		if (!punct(p, "(")) {
			error = "CREATE " + obj + ": expected '(' with the arguments";
			break;
		}
		if (!if_not_exists) {
			// Fast path: swap the keyword span for the function name, leaving the
			// `(...)` argument list (and the rest of the batch) untouched.
			cur = cur.substr(0, toks[found].start) + "SELECT create_" + obj + cur.substr(toks[kw].end);
			rewrites++;
			continue;
		}
		// IF NOT EXISTS: drop those words and append a trailing `true` flag argument
		// just before the closing paren, so create_x(...) sees the keep-existing bit.
		int depth = 0;
		size_t close = 0;
		bool balanced = false;
		for (size_t q = p; q < toks.size(); q++) {
			if (punct(q, "(")) {
				depth++;
			} else if (punct(q, ")")) {
				if (--depth == 0) {
					close = q;
					balanced = true;
					break;
				}
			}
		}
		if (!balanced) {
			error = "CREATE " + obj + ": unbalanced parentheses";
			break;
		}
		std::string inner = cur.substr(toks[p].end, toks[close].start - toks[p].end);
		cur = cur.substr(0, toks[found].start) + "SELECT create_" + obj + "(" + inner + ", true)" +
		      cur.substr(toks[close].end);
		rewrites++;
	}

	// ASK('question') ON (t1, t2) RETURN (col type, ...)  [and its ask_sql sibling]
	//   -> ask('question', 't1, t2', 'col type, ...')     (a table function call)
	// Only the full keyword shape is rewritten; a bare ask(...) call is skipped, so
	// the loop terminates.
	for (int guard = 0; guard < 32; guard++) {
		auto toks = SjLex(cur);
		auto lw = [&](size_t i) {
			return (i < toks.size() && toks[i].is_word) ? toks[i].lword : std::string();
		};
		auto punct = [&](size_t i, const char *p) {
			return i < toks.size() && !toks[i].is_word && !toks[i].is_string &&
			       cur.compare(toks[i].start, toks[i].end - toks[i].start, p) == 0;
		};
		auto find_close = [&](size_t open) -> int {
			int d = 0;
			for (size_t q = open; q < toks.size(); q++) {
				if (punct(q, "(")) {
					d++;
				} else if (punct(q, ")") && --d == 0) {
					return (int)q;
				}
			}
			return -1;
		};
		auto inner = [&](size_t open, size_t close) {
			std::string raw = cur.substr(toks[open].end, toks[close].start - toks[open].end);
			StringUtil::Trim(raw);
			return raw;
		};

		bool rewrote = false;
		for (size_t j = 0; j + 1 < toks.size(); j++) {
			if ((lw(j) != "ask" && lw(j) != "ask_sql") || !punct(j + 1, "(")) {
				continue;
			}
			std::string fn = lw(j);
			int c1 = find_close(j + 1);
			if (c1 < 0) {
				continue;
			}
			// Optional ` ON (...)` then a required ` RETURN (...)` tail; otherwise it's a
			// plain call. Without ON, the table list is empty and the scope must come from
			// ask_dbt_scope_tags (a domain-scoped ASK).
			size_t after = (size_t)c1 + 1;
			std::string tables;
			size_t ret_i;
			if (lw(after) == "on" && punct(after + 1, "(")) {
				int c2 = find_close(after + 1);
				if (c2 < 0) {
					continue;
				}
				tables = inner(after + 1, (size_t)c2);
				ret_i = (size_t)c2 + 1;
			} else {
				ret_i = after;
			}
			if (lw(ret_i) != "return" || !punct(ret_i + 1, "(")) {
				continue;
			}
			int c3 = find_close(ret_i + 1);
			if (c3 < 0) {
				continue;
			}
			std::string req = inner(j + 1, (size_t)c1); // already a quoted string literal
			std::string schema = inner(ret_i + 1, (size_t)c3);
			std::string repl = fn + "(" + req + ", " + SqlQuote(tables) + ", " + SqlQuote(schema) + ")";
			// Allow the bare `ASK(...) ON ... RETURN ...` form (no SELECT/FROM): when the
			// keyword opens the statement, prepend FROM so it's a valid table reference.
			if (j == 0 || punct(j - 1, ";")) {
				repl = "FROM " + repl;
			}
			cur = cur.substr(0, toks[j].start) + repl + cur.substr(toks[c3].end);
			rewrites++;
			rewrote = true;
			break; // re-tokenize for any further ASK statements
		}
		if (!rewrote) {
			break;
		}
	}

	if (!error.empty()) {
		ParserException e(error);
		return ParserOverrideResult(e);
	}
	if (rewrites == 0) {
		return ParserOverrideResult(); // not ours — let DuckDB's original error stand
	}
	return ReparseAs(cur, options);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
static void LoadInternal(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();
	auto &config = DBConfig::GetConfig(db);

	config.AddExtensionOption("duckthink_log",
	                          "Emit a one-line stderr trace per LLM call (provider:model, tokens, ms). "
	                          "Off by default; metrics are always collected — see duckthink_metrics().",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(false));
	config.AddExtensionOption("ask_max_retries",
	                          "ASK() self-correction: how many times to feed a failing generated query's "
	                          "error back to the model before giving up (0 = one shot). Default 2.",
	                          LogicalType::BIGINT, Value::BIGINT(2));
	config.AddExtensionOption("ask_dbt_semantic_manifest",
	                          "Path to a dbt Semantic Layer manifest (target/semantic_manifest.json) and/or a "
	                          "target/manifest.json. When set, ASK() grounds the prompt in the semantic models "
	                          "(entities, dimensions, measures), metric definitions, and column descriptions for "
	                          "the tables in scope — closing the text-to-SQL accuracy gap. Empty = off.",
	                          LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("ask_dbt_scope_tags",
	                          "Comma-separated dbt tags. When set (with ask_dbt_semantic_manifest), ASK() scopes to "
	                          "every model/source carrying any of these tags — so a service names its DOMAIN "
	                          "('taxi') instead of listing tables in ON (...). Unioned with any explicit ON "
	                          "tables. Empty = off.",
	                          LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("ask_scope_max_tables",
	                          "When a scope resolves to more than this many tables (e.g. a whole tagged domain), "
	                          "ASK() narrows it to the tables relevant to the question before building the prompt. "
	                          "0 disables the step. Default 12.",
	                          LogicalType::BIGINT, Value::BIGINT(12));
	config.AddExtensionOption("ask_embed_model",
	                          "Registered embedding model (CREATE MODEL, provider openai/ollama). When set, table "
	                          "selection runs in two stages: a cheap EMBEDDING recall ranks all candidates by cosine "
	                          "to the question (summaries embedded once and cached) down to ask_retrieve_k, then the "
	                          "LLM reranks. Scales to very large domains. Empty = LLM routing only.",
	                          LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("ask_retrieve_k",
	                          "How many candidate tables the embedding recall keeps before the LLM rerank. Only used "
	                          "when ask_embed_model is set. Default 30.",
	                          LogicalType::BIGINT, Value::BIGINT(30));

	// LLM plumbing (provider secrets, model registry, metrics) + the ASK functions.
	RegisterAiFunctions(loader);

	// Custom keyword syntax (ASK ... ON ... RETURN, CREATE/DROP MODEL) via a
	// parser_override that only fires on FALLBACK, so it never shadows valid SQL.
	ParserExtension pe;
	pe.parser_override = DuckthinkParserOverride;
	ParserExtension::Register(config, pe);

	// Enable parser overrides on load so the keyword forms work without a manual
	// SET. FALLBACK is safe: any query we don't claim is parsed by DuckDB normally.
	try {
		config.SetOptionByName("allow_parser_override_extension", Value("FALLBACK"));
	} catch (...) {
		// older DuckDB without this option: user can SET it manually
	}
}

void DuckthinkExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string DuckthinkExtension::Name() {
	return "duckthink";
}
std::string DuckthinkExtension::Version() const {
#ifdef EXT_VERSION_DUCKTHINK
	return EXT_VERSION_DUCKTHINK;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(duckthink, loader) {
	duckdb::LoadInternal(loader);
}
}

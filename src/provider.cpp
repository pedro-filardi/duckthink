//===----------------------------------------------------------------------===//
// llm.cpp — model registry + OpenAI-compatible chat transport.
//
// No DuckDB headers here on purpose: this is the only translation unit that
// pulls in httplib, keeping its many macros away from the DuckDB sources.
//===----------------------------------------------------------------------===//
#include "provider.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <chrono>
#include <sstream>
#include <thread>
#include <stdexcept>

// When built with TLS (-DDUCKTHINK_TLS, set by CMake when OpenSSL is found), turn
// on httplib's OpenSSL support so https:// providers (OpenAI) work directly. DuckDB's
// httplib selects a DIFFERENT namespace under the OpenSSL macro
// (duckdb_httplib_openssl vs duckdb_httplib), so this translation unit's symbols never
// clash with DuckDB's own no-TLS copy — no ODR risk.
#ifdef DUCKTHINK_TLS
#define CPPHTTPLIB_OPENSSL_SUPPORT
#endif
#include "httplib.hpp"
#ifdef DUCKTHINK_TLS
namespace httplib = duckdb_httplib_openssl;
#else
namespace httplib = duckdb_httplib;
#endif

#include "yyjson.hpp"

namespace duckthink {

namespace yj = duckdb_yyjson;

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------
namespace {

std::string Lower(const std::string &s) {
	std::string r(s);
	std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) { return std::tolower(c); });
	return r;
}

std::mutex &RegistryMutex() {
	static std::mutex m;
	return m;
}

std::map<std::string, LlmModel> &Registry() {
	static std::map<std::string, LlmModel> reg; // keyed by lowercased name
	return reg;
}

} // namespace

void RegisterModel(const LlmModel &model) {
	std::lock_guard<std::mutex> guard(RegistryMutex());
	Registry()[Lower(model.name)] = model;
}

bool LookupModel(const std::string &name, LlmModel &out) {
	std::lock_guard<std::mutex> guard(RegistryMutex());
	auto it = Registry().find(Lower(name));
	if (it == Registry().end()) {
		return false;
	}
	out = it->second;
	return true;
}

bool DropModel(const std::string &name) {
	std::lock_guard<std::mutex> guard(RegistryMutex());
	return Registry().erase(Lower(name)) > 0;
}

std::vector<LlmModel> AllModels() {
	std::lock_guard<std::mutex> guard(RegistryMutex());
	std::vector<LlmModel> out;
	out.reserve(Registry().size());
	for (auto &kv : Registry()) {
		out.push_back(kv.second);
	}
	return out;
}

namespace {
std::map<std::string, LlmPrompt> &PromptRegistry() {
	static std::map<std::string, LlmPrompt> reg; // keyed by lowercased name
	return reg;
}
} // namespace

void RegisterPrompt(const LlmPrompt &prompt) {
	std::lock_guard<std::mutex> guard(RegistryMutex());
	PromptRegistry()[Lower(prompt.name)] = prompt;
}

bool LookupPrompt(const std::string &name, LlmPrompt &out) {
	std::lock_guard<std::mutex> guard(RegistryMutex());
	auto it = PromptRegistry().find(Lower(name));
	if (it == PromptRegistry().end()) {
		return false;
	}
	out = it->second;
	return true;
}

bool DropPrompt(const std::string &name) {
	std::lock_guard<std::mutex> guard(RegistryMutex());
	return PromptRegistry().erase(Lower(name)) > 0;
}

std::vector<LlmPrompt> AllPrompts() {
	std::lock_guard<std::mutex> guard(RegistryMutex());
	std::vector<LlmPrompt> out;
	out.reserve(PromptRegistry().size());
	for (auto &kv : PromptRegistry()) {
		out.push_back(kv.second);
	}
	return out;
}

// --- Observability accumulator (process-wide, mutex-guarded) ----------------
namespace {
std::mutex &StatsMutex() {
	static std::mutex m;
	return m;
}
std::map<std::string, LlmStat> &LlmStatsMap() {
	static std::map<std::string, LlmStat> s; // keyed by "provider:model_id"
	return s;
}
EmbedStat &EmbedStatsRef() {
	static EmbedStat e;
	return e;
}
} // namespace

void RecordLlmCall(const std::string &provider, const std::string &model_id, uint64_t input_tokens,
                   uint64_t output_tokens, uint64_t retries, uint64_t rate_limited_ms, uint64_t wall_ms, bool error) {
	std::string key = provider + ":" + model_id;
	std::lock_guard<std::mutex> guard(StatsMutex());
	auto &s = LlmStatsMap()[key];
	s.model = key;
	s.api_calls++;
	s.input_tokens += input_tokens;
	s.output_tokens += output_tokens;
	s.retries += retries;
	s.rate_limited_ms += rate_limited_ms;
	s.wall_ms += wall_ms;
	if (error) {
		s.errors++;
	}
}

void RecordCacheHit(const std::string &provider, const std::string &model_id) {
	std::string key = provider + ":" + model_id;
	std::lock_guard<std::mutex> guard(StatsMutex());
	auto &s = LlmStatsMap()[key];
	s.model = key;
	s.cache_hits++;
}

void RecordEmbed(uint64_t texts, uint64_t wall_ms) {
	std::lock_guard<std::mutex> guard(StatsMutex());
	EmbedStatsRef().texts += texts;
	EmbedStatsRef().wall_ms += wall_ms;
}

std::vector<LlmStat> SnapshotLlmStats() {
	std::lock_guard<std::mutex> guard(StatsMutex());
	std::vector<LlmStat> out;
	out.reserve(LlmStatsMap().size());
	for (auto &kv : LlmStatsMap()) {
		out.push_back(kv.second);
	}
	return out;
}

EmbedStat SnapshotEmbedStats() {
	std::lock_guard<std::mutex> guard(StatsMutex());
	return EmbedStatsRef();
}

void ResetStats() {
	std::lock_guard<std::mutex> guard(StatsMutex());
	LlmStatsMap().clear();
	EmbedStatsRef() = EmbedStat {};
}

std::string DefaultBaseUrl(const std::string &provider) {
	std::string p = Lower(provider);
	if (p == "ollama") {
		return "http://localhost:11434";
	}
	if (p == "openai") {
		return "https://api.openai.com";
	}
	if (p == "anthropic") {
		return "https://api.anthropic.com";
	}
	// Unknown provider: no sensible default; caller must supply endpoint.
	return std::string();
}

// ---------------------------------------------------------------------------
// Minimal, correct JSON helpers (request build + response extraction)
// ---------------------------------------------------------------------------
namespace {

void JsonEscapeTo(std::string &out, const std::string &s) {
	for (unsigned char c : s) {
		switch (c) {
		case '"':
			out += "\\\"";
			break;
		case '\\':
			out += "\\\\";
			break;
		case '\b':
			out += "\\b";
			break;
		case '\f':
			out += "\\f";
			break;
		case '\n':
			out += "\\n";
			break;
		case '\r':
			out += "\\r";
			break;
		case '\t':
			out += "\\t";
			break;
		default:
			if (c < 0x20) {
				static const char *hex = "0123456789abcdef";
				out += "\\u00";
				out += hex[(c >> 4) & 0xF];
				out += hex[c & 0xF];
			} else {
				out += static_cast<char>(c);
			}
		}
	}
}

// Read a JSON string token starting at body[pos] == '"'. Advances pos past the
// closing quote and appends the decoded (unescaped) content to out. Returns
// false on a malformed / unterminated string.
bool ReadJsonString(const std::string &body, size_t &pos, std::string &out) {
	if (pos >= body.size() || body[pos] != '"') {
		return false;
	}
	++pos; // skip opening quote
	while (pos < body.size()) {
		char c = body[pos++];
		if (c == '"') {
			return true;
		}
		if (c != '\\') {
			out += c;
			continue;
		}
		if (pos >= body.size()) {
			return false;
		}
		char esc = body[pos++];
		switch (esc) {
		case '"':
			out += '"';
			break;
		case '\\':
			out += '\\';
			break;
		case '/':
			out += '/';
			break;
		case 'b':
			out += '\b';
			break;
		case 'f':
			out += '\f';
			break;
		case 'n':
			out += '\n';
			break;
		case 'r':
			out += '\r';
			break;
		case 't':
			out += '\t';
			break;
		case 'u': {
			if (pos + 4 > body.size()) {
				return false;
			}
			auto hexval = [](char h) -> int {
				if (h >= '0' && h <= '9')
					return h - '0';
				if (h >= 'a' && h <= 'f')
					return h - 'a' + 10;
				if (h >= 'A' && h <= 'F')
					return h - 'A' + 10;
				return -1;
			};
			int cp = 0;
			for (int i = 0; i < 4; i++) {
				int v = hexval(body[pos + i]);
				if (v < 0) {
					return false;
				}
				cp = (cp << 4) | v;
			}
			pos += 4;
			// Surrogate pair -> combine into a code point.
			if (cp >= 0xD800 && cp <= 0xDBFF && pos + 6 <= body.size() && body[pos] == '\\' && body[pos + 1] == 'u') {
				int lo = 0;
				bool ok = true;
				for (int i = 0; i < 4; i++) {
					int v = hexval(body[pos + 2 + i]);
					if (v < 0) {
						ok = false;
						break;
					}
					lo = (lo << 4) | v;
				}
				if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
					cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
					pos += 6;
				}
			}
			// Encode the code point as UTF-8.
			if (cp <= 0x7F) {
				out += static_cast<char>(cp);
			} else if (cp <= 0x7FF) {
				out += static_cast<char>(0xC0 | (cp >> 6));
				out += static_cast<char>(0x80 | (cp & 0x3F));
			} else if (cp <= 0xFFFF) {
				out += static_cast<char>(0xE0 | (cp >> 12));
				out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
				out += static_cast<char>(0x80 | (cp & 0x3F));
			} else {
				out += static_cast<char>(0xF0 | (cp >> 18));
				out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
				out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
				out += static_cast<char>(0x80 | (cp & 0x3F));
			}
			break;
		}
		default:
			// Unknown escape: keep the character verbatim.
			out += esc;
		}
	}
	return false; // unterminated
}

// Find the first occurrence of the object key "<key>" and, if it is followed by
// a string value, decode it into out. Returns false if the key is absent or its
// value is not a string. `from` lets the caller search past an earlier key
// (e.g. skip "role" to reach the assistant "content").
bool FindStringValue(const std::string &body, const std::string &key, std::string &out, size_t from = 0) {
	const std::string needle = "\"" + key + "\"";
	size_t k = body.find(needle, from);
	while (k != std::string::npos) {
		size_t p = k + needle.size();
		while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) {
			p++;
		}
		if (p < body.size() && body[p] == ':') {
			p++;
			while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) {
				p++;
			}
			if (p < body.size() && body[p] == '"') {
				out.clear();
				return ReadJsonString(body, p, out);
			}
			return false; // key present but value isn't a string
		}
		k = body.find(needle, k + needle.size());
	}
	return false;
}

// Parse "scheme://host[:port]" out of a base_url. Returns false if it isn't a
// well-formed http(s) URL.
bool SplitBaseUrl(const std::string &url, std::string &scheme, std::string &host, int &port) {
	auto sep = url.find("://");
	if (sep == std::string::npos) {
		return false;
	}
	scheme = Lower(url.substr(0, sep));
	std::string rest = url.substr(sep + 3);
	// strip any path
	auto slash = rest.find('/');
	if (slash != std::string::npos) {
		rest = rest.substr(0, slash);
	}
	auto colon = rest.find(':');
	if (colon != std::string::npos) {
		host = rest.substr(0, colon);
		try {
			port = std::stoi(rest.substr(colon + 1));
		} catch (...) {
			return false;
		}
	} else {
		host = rest;
		port = (scheme == "https") ? 443 : 80;
	}
	return !host.empty() && (scheme == "http" || scheme == "https");
}

// Like SplitBaseUrl but also returns the path (with leading '/'). Used by the
// model downloader, which needs the full request path, not just the origin.
bool SplitUrl(const std::string &url, std::string &scheme, std::string &host, int &port, std::string &path) {
	auto sep = url.find("://");
	if (sep == std::string::npos) {
		return false;
	}
	scheme = Lower(url.substr(0, sep));
	std::string rest = url.substr(sep + 3);
	auto slash = rest.find('/');
	std::string hostport;
	if (slash != std::string::npos) {
		hostport = rest.substr(0, slash);
		path = rest.substr(slash);
	} else {
		hostport = rest;
		path = "/";
	}
	auto colon = hostport.find(':');
	if (colon != std::string::npos) {
		host = hostport.substr(0, colon);
		try {
			port = std::stoi(hostport.substr(colon + 1));
		} catch (...) {
			return false;
		}
	} else {
		host = hostport;
		port = (scheme == "https") ? 443 : 80;
	}
	return !host.empty() && (scheme == "http" || scheme == "https");
}

// Advance `pos` to just after the colon following the first `"key"`. Returns
// false if the key isn't found.
bool SeekKeyValue(const std::string &body, const std::string &key, size_t &pos) {
	const std::string needle = "\"" + key + "\"";
	size_t k = body.find(needle);
	if (k == std::string::npos) {
		return false;
	}
	size_t p = k + needle.size();
	while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) {
		p++;
	}
	if (p >= body.size() || body[p] != ':') {
		return false;
	}
	p++;
	while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) {
		p++;
	}
	pos = p;
	return true;
}

// How long to wait after a 429 before retrying. Prefers the server's own
// guidance so we don't over- or under-wait: the `Retry-After` header (seconds),
// else the "Please try again in 473ms / 1.5s" hint OpenAI puts in the body.
// Falls back to a growing backoff. A small margin + per-attempt escalation keeps
// several concurrent callers from all waking into the same still-saturated
// window (TPM resets over a rolling 60s, so escalating eventually drains it).
double RetryWaitSeconds(const std::string &retry_after_hdr, const std::string &body, int rl_attempt) {
	double base = -1.0;
	if (!retry_after_hdr.empty()) {
		try {
			double v = std::stod(retry_after_hdr);
			if (v >= 0) {
				base = v;
			}
		} catch (...) {
		}
	}
	if (base < 0) {
		auto p = body.find("try again in ");
		if (p != std::string::npos) {
			p += 13; // strlen("try again in ")
			size_t q = p;
			while (q < body.size() && (std::isdigit(static_cast<unsigned char>(body[q])) || body[q] == '.')) {
				q++;
			}
			if (q > p) {
				try {
					double num = std::stod(body.substr(p, q - p));
					base = (body.compare(q, 2, "ms") == 0) ? num / 1000.0 : num; // else seconds
				} catch (...) {
				}
			}
		}
	}
	if (base < 0) {
		base = 0.5 * static_cast<double>(1 << std::min(rl_attempt, 5)); // 0.5,1,2,4,8,16
	}
	double wait = base + 0.25 + 0.5 * rl_attempt; // margin + escalation
	if (wait > 30.0) {
		wait = 30.0;
	}
	return wait;
}

} // namespace

bool DownloadToFile(const std::string &url_in, const std::string &dest_path, std::string &err) {
	std::string url = url_in;
	const std::string tmp = dest_path + ".part";
	// Follow redirects manually (HF resolve/ 302s to a signed CDN URL, often on a
	// different host) rather than relying on httplib's cross-host follow.
	for (int hop = 0; hop < 8; hop++) {
		std::string scheme, host, path;
		int port = 0;
		if (!SplitUrl(url, scheme, host, port, path)) {
			err = "invalid URL: " + url;
			return false;
		}
#ifndef DUCKTHINK_TLS
		if (scheme == "https") {
			err = "https download needs a TLS build (OpenSSL); run scripts/fetch_models.sh instead";
			return false;
		}
#endif
		std::string origin = scheme + "://" + host;
		if (!((scheme == "http" && port == 80) || (scheme == "https" && port == 443))) {
			origin += ":" + std::to_string(port);
		}
		httplib::Client cli(origin);
		cli.set_connection_timeout(30, 0);
		cli.set_read_timeout(600, 0); // large files, slow mirrors

		int status = -1;
		std::string location;
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		if (!out) {
			err = "cannot write to " + tmp;
			return false;
		}
		auto res = cli.Get(
		    path, httplib::Headers {{"User-Agent", "duckthink"}},
		    [&](const httplib::Response &r) {
			    status = r.status;
			    if (r.has_header("Location")) {
				    location = r.get_header_value("Location");
			    }
			    return true; // proceed to the body
		    },
		    [&](const char *data, size_t len) {
			    if (status == 200) { // don't persist redirect/error bodies
				    out.write(data, (std::streamsize)len);
			    }
			    return static_cast<bool>(out);
		    });
		out.close();
		if (!res) {
			std::remove(tmp.c_str());
			err = "connection failed to " + host;
			return false;
		}
		if (status >= 300 && status < 400 && !location.empty()) {
			if (location.find("://") == std::string::npos) { // relative redirect
				if (location.empty() || location[0] != '/') {
					location = "/" + location;
				}
				location = origin + location;
			}
			url = location;
			continue;
		}
		if (status != 200) {
			std::remove(tmp.c_str());
			err = "HTTP " + std::to_string(status) + " downloading " + url;
			return false;
		}
		std::remove(dest_path.c_str());
		if (std::rename(tmp.c_str(), dest_path.c_str()) != 0) {
			err = "could not move download into place at " + dest_path;
			return false;
		}
		return true;
	}
	std::remove(tmp.c_str());
	err = "too many redirects for " + url_in;
	return false;
}

bool JsonFindBool(const std::string &body, const std::string &key, bool &out) {
	size_t p;
	if (!SeekKeyValue(body, key, p)) {
		return false;
	}
	if (body.compare(p, 4, "true") == 0) {
		out = true;
		return true;
	}
	if (body.compare(p, 5, "false") == 0) {
		out = false;
		return true;
	}
	// Tolerate quoted "true"/"false" or 1/0.
	if (p < body.size() && (body[p] == '1' || (body[p] == '"' && body.compare(p, 6, "\"true\"") == 0))) {
		out = true;
		return true;
	}
	if (p < body.size() && (body[p] == '0' || (body[p] == '"' && body.compare(p, 7, "\"false\"") == 0))) {
		out = false;
		return true;
	}
	return false;
}

bool JsonFindInt(const std::string &body, const std::string &key, long &out) {
	size_t p;
	if (!SeekKeyValue(body, key, p)) {
		return false;
	}
	if (p < body.size() && body[p] == '"') {
		p++; // tolerate a quoted number
	}
	size_t start = p;
	if (p < body.size() && (body[p] == '-' || body[p] == '+')) {
		p++;
	}
	size_t digits = p;
	while (p < body.size() && std::isdigit(static_cast<unsigned char>(body[p]))) {
		p++;
	}
	if (p == digits) {
		return false; // no digits
	}
	try {
		out = std::stol(body.substr(start, p - start));
	} catch (...) {
		return false;
	}
	return true;
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------
std::string LlmChat(const LlmCall &call, const std::string &system_prompt, const std::string &user_prompt) {
	if (call.model_id.empty()) {
		throw std::runtime_error("llm: model id is empty (check CREATE MODEL)");
	}
	std::string base_url = call.base_url.empty() ? DefaultBaseUrl(call.provider) : call.base_url;
	if (base_url.empty()) {
		throw std::runtime_error("llm: no endpoint for provider '" + call.provider +
		                         "' (set 'endpoint' in the secret, e.g. CREATE SECRET (TYPE " + call.provider +
		                         ", ENDPOINT 'http://host:port'))");
	}

	std::string scheme, host;
	int port = 0;
	if (!SplitBaseUrl(base_url, scheme, host, port)) {
		throw std::runtime_error("llm: malformed endpoint '" + base_url + "'");
	}
#ifndef DUCKTHINK_TLS
	if (scheme == "https") {
		// This binary was compiled without OpenSSL, so only plaintext HTTP works.
		// Local providers (Ollama, llama.cpp, LM Studio, vLLM) all serve HTTP.
		throw std::runtime_error(
		    "llm: HTTPS endpoint (" + base_url +
		    ") requires a TLS build. This binary was compiled without OpenSSL — rebuild with OpenSSL "
		    "available (see README), use a local HTTP provider such as Ollama, or the openai_proxy.");
	}
#endif

	// Anthropic's native Messages API is not OpenAI-shaped: /v1/messages, x-api-key
	// auth, system as a top-level field, max_tokens required, response in content[].
	const bool is_anthropic = Lower(call.provider) == "anthropic";

	std::string body;
	body.reserve(user_prompt.size() + system_prompt.size() + 128);
	if (is_anthropic) {
		body += "{\"model\":\"";
		JsonEscapeTo(body, call.model_id);
		body += "\",\"max_tokens\":" + std::to_string(call.max_tokens);
		if (!system_prompt.empty()) {
			body += ",\"system\":\"";
			JsonEscapeTo(body, system_prompt);
			body += "\"";
		}
		body += ",\"messages\":[{\"role\":\"user\",\"content\":\"";
		JsonEscapeTo(body, user_prompt);
		body += "\"}]";
		std::ostringstream t;
		t << call.temperature;
		body += ",\"temperature\":" + t.str() + "}";
	} else {
		// OpenAI-compatible chat body (OpenAI, Ollama, Anthropic's compat endpoint, ...).
		body += "{\"model\":\"";
		JsonEscapeTo(body, call.model_id);
		body += "\",\"messages\":[";
		if (!system_prompt.empty()) {
			body += "{\"role\":\"system\",\"content\":\"";
			JsonEscapeTo(body, system_prompt);
			body += "\"},";
		}
		body += "{\"role\":\"user\",\"content\":\"";
		JsonEscapeTo(body, user_prompt);
		body += "\"}]";
		std::ostringstream t;
		t << call.temperature;
		body += ",\"temperature\":" + t.str();
		if (!call.response_format.empty()) {
			body += ",\"response_format\":" + call.response_format;
		}
		body += ",\"stream\":false}";
	}

	// Construct from the origin URL so httplib auto-selects an SSL client for https
	// (when compiled with OpenSSL); plain HTTP otherwise.
	std::string origin = scheme + "://" + host;
	if (!((scheme == "http" && port == 80) || (scheme == "https" && port == 443))) {
		origin += ":" + std::to_string(port);
	}
	httplib::Client cli(origin);
	cli.set_connection_timeout(call.timeout_seconds, 0);
	cli.set_read_timeout(call.timeout_seconds, 0);
	cli.set_write_timeout(call.timeout_seconds, 0);

	httplib::Headers headers;
	// Trim stray whitespace/newlines from the key so a copy-paste slip can't
	// corrupt the HTTP header framing.
	std::string api_key = call.api_key;
	api_key.erase(0, api_key.find_first_not_of(" \t\r\n"));
	size_t key_end = api_key.find_last_not_of(" \t\r\n");
	if (key_end != std::string::npos) {
		api_key.erase(key_end + 1);
	}
	if (is_anthropic) {
		headers.emplace("x-api-key", api_key);
		headers.emplace("anthropic-version", "2023-06-01");
	} else if (!api_key.empty()) {
		headers.emplace("Authorization", "Bearer " + api_key);
	}

	// Retry policy:
	//  - connection errors / 5xx: transient, exponential backoff, bounded by max_retries.
	//  - 429 (rate limit): NOT a failure — wait out the server's Retry-After and keep
	//    going, bounded only by a generous total-time budget (rate_limit_wait_seconds).
	//    These waits don't consume the transient-error budget.
	//  - any other 4xx: real client error, fail fast.
	std::string last_err;
	int transient_left = call.max_retries < 0 ? 0 : call.max_retries;
	int transient_attempt = 0; // drives 5xx/connection backoff
	int rl_attempt = 0;        // drives 429 wait escalation
	double rl_budget = call.rate_limit_wait_seconds > 0 ? call.rate_limit_wait_seconds : 0.0;
	const char *endpoint = is_anthropic ? "/v1/messages" : "/v1/chat/completions";

	// Observability: time the call, sum any 429 wait, and make sure every exit is
	// counted exactly once — success records tokens explicitly; any throw path is
	// caught by the finalizer and booked as an error.
	auto call_t0 = std::chrono::steady_clock::now();
	uint64_t rate_limited_ms = 0;
	bool recorded = false;
	auto elapsed_ms = [&]() -> uint64_t {
		return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
		                                                                       call_t0)
		    .count();
	};
	struct Finalizer {
		std::function<void()> fn;
		~Finalizer() {
			if (fn) {
				fn();
			}
		}
	} finalize;
	finalize.fn = [&]() {
		if (!recorded) {
			RecordLlmCall(call.provider, call.model_id, 0, 0, (uint64_t)transient_attempt, rate_limited_ms,
			              elapsed_ms(), true);
		}
	};

	for (;;) {
		auto res = cli.Post(endpoint, headers, body, "application/json");
		if (!res) {
			last_err = "cannot reach " + base_url + " (" + httplib::to_string(res.error()) +
			           "). Is the provider running? For Ollama: `ollama serve`.";
			if (transient_left-- <= 0) {
				break;
			}
			int backoff_ms = 400 * (1 << std::min(transient_attempt++, 5)); // 400,800,...,12800 capped
			std::this_thread::sleep_for(std::chrono::milliseconds(std::min(backoff_ms, 8000)));
			continue;
		}
		if (res->status == 429) {
			std::string msg;
			FindStringValue(res->body, "message", msg);
			if (rl_budget <= 0.0) {
				last_err = "provider returned HTTP 429 (rate limit)" + (msg.empty() ? "" : ": " + msg) +
				           " — exhausted the rate-limit wait budget; lower llm_join_concurrency or raise "
				           "llm_rate_limit_wait";
				break;
			}
			double wait_s = RetryWaitSeconds(res->get_header_value("Retry-After"), res->body, rl_attempt++);
			if (wait_s > rl_budget) {
				wait_s = rl_budget; // one last wait up to the remaining budget
			}
			rl_budget -= wait_s;
			rate_limited_ms += (uint64_t)(wait_s * 1000.0);
			std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<long>(wait_s * 1000.0)));
			continue; // rate limit -> keep going, don't spend the transient budget
		}
		if (res->status >= 500) {
			std::string msg;
			FindStringValue(res->body, "message", msg);
			last_err = "provider returned HTTP " + std::to_string(res->status) + (msg.empty() ? "" : ": " + msg);
			if (transient_left-- <= 0) {
				break;
			}
			int backoff_ms = 400 * (1 << std::min(transient_attempt++, 5));
			std::this_thread::sleep_for(std::chrono::milliseconds(std::min(backoff_ms, 8000)));
			continue; // transient -> retry
		}
		if (res->status < 200 || res->status >= 300) {
			std::string msg;
			if (FindStringValue(res->body, "message", msg) || FindStringValue(res->body, "error", msg)) {
				throw std::runtime_error("llm: provider returned HTTP " + std::to_string(res->status) + ": " + msg);
			}
			throw std::runtime_error("llm: provider returned HTTP " + std::to_string(res->status) + ": " + res->body);
		}
		// 2xx: pull out the assistant text.
		std::string content;
		if (is_anthropic) {
			// Anthropic: {"content":[{"type":"text","text":"..."}], ...}. FindStringValue's
			// colon check skips the "type":"text" value and lands on the "text": key.
			if (!FindStringValue(res->body, "text", content)) {
				throw std::runtime_error("llm: could not parse completion from response: " + res->body.substr(0, 400));
			}
		} else {
			// OpenAI-shape: skip the leading "role":"assistant" to reach the message content.
			size_t role_pos = res->body.find("\"role\"");
			if (!FindStringValue(res->body, "content", content, role_pos == std::string::npos ? 0 : role_pos)) {
				throw std::runtime_error("llm: could not parse completion from response: " + res->body.substr(0, 400));
			}
		}

		// Token usage: Anthropic reports input_tokens/output_tokens, OpenAI/Ollama
		// prompt_tokens/completion_tokens. Absent (some local servers) -> 0.
		long in_tok = 0, out_tok = 0;
		if (is_anthropic) {
			JsonFindInt(res->body, "input_tokens", in_tok);
			JsonFindInt(res->body, "output_tokens", out_tok);
		} else {
			JsonFindInt(res->body, "prompt_tokens", in_tok);
			JsonFindInt(res->body, "completion_tokens", out_tok);
		}
		uint64_t wall_ms = elapsed_ms();
		recorded = true;
		RecordLlmCall(call.provider, call.model_id, (uint64_t)(in_tok < 0 ? 0 : in_tok),
		              (uint64_t)(out_tok < 0 ? 0 : out_tok), (uint64_t)transient_attempt, rate_limited_ms, wall_ms,
		              false);
		if (call.log) {
			fprintf(stderr, "[duckthink] %s:%s  %ld+%ld tok  %llums%s\n", call.provider.c_str(), call.model_id.c_str(),
			        in_tok, out_tok, (unsigned long long)wall_ms, transient_attempt ? "  (retried)" : "");
		}
		return content;
	}
	throw std::runtime_error("llm: " + last_err);
}

// Embed a batch of texts via the provider's OpenAI-compatible /v1/embeddings endpoint
// (OpenAI and Ollama both serve it). Returns one vector per input, in input order.
// Anthropic has no embeddings endpoint. Requests are chunked to stay within provider
// batch limits; connection/5xx/429 are retried like chat.
std::vector<std::vector<float>> LlmEmbed(const LlmCall &call, const std::vector<std::string> &texts) {
	std::vector<std::vector<float>> out;
	if (texts.empty()) {
		return out;
	}
	if (Lower(call.provider) == "anthropic") {
		throw std::runtime_error("embed: provider 'anthropic' has no embeddings endpoint — set ask_embed_model to an "
		                         "openai or ollama embedding model (e.g. text-embedding-3-small)");
	}
	std::string base_url = call.base_url.empty() ? DefaultBaseUrl(call.provider) : call.base_url;
	std::string scheme, host;
	int port = 0;
	if (base_url.empty() || !SplitBaseUrl(base_url, scheme, host, port)) {
		throw std::runtime_error("embed: malformed or missing endpoint '" + base_url + "'");
	}
#ifndef DUCKTHINK_TLS
	if (scheme == "https") {
		throw std::runtime_error("embed: HTTPS endpoint requires a TLS build (rebuild with OpenSSL).");
	}
#endif
	std::string origin = scheme + "://" + host;
	if (!((scheme == "http" && port == 80) || (scheme == "https" && port == 443))) {
		origin += ":" + std::to_string(port);
	}
	httplib::Client cli(origin);
	cli.set_connection_timeout(call.timeout_seconds, 0);
	cli.set_read_timeout(call.timeout_seconds, 0);
	cli.set_write_timeout(call.timeout_seconds, 0);
	httplib::Headers headers;
	std::string api_key = call.api_key;
	api_key.erase(0, api_key.find_first_not_of(" \t\r\n"));
	size_t key_end = api_key.find_last_not_of(" \t\r\n");
	if (key_end != std::string::npos) {
		api_key.erase(key_end + 1);
	}
	if (!api_key.empty()) {
		headers.emplace("Authorization", "Bearer " + api_key);
	}

	auto t0 = std::chrono::steady_clock::now();
	const size_t kChunk = 128;
	for (size_t start = 0; start < texts.size(); start += kChunk) {
		size_t end = std::min(start + kChunk, texts.size());
		std::string body = "{\"model\":\"";
		JsonEscapeTo(body, call.model_id);
		body += "\",\"input\":[";
		for (size_t i = start; i < end; i++) {
			if (i > start) {
				body += ",";
			}
			body += "\"";
			JsonEscapeTo(body, texts[i]);
			body += "\"";
		}
		body += "]}";

		std::string last_err;
		int transient_left = call.max_retries < 0 ? 0 : call.max_retries;
		int attempt = 0;
		httplib::Result res(nullptr, httplib::Error::Unknown);
		for (;;) {
			res = cli.Post("/v1/embeddings", headers, body, "application/json");
			if (!res || res->status == 429 || res->status >= 500) {
				last_err = res ? ("HTTP " + std::to_string(res->status))
				               : ("cannot reach " + base_url + " (" + httplib::to_string(res.error()) + ")");
				if (transient_left-- <= 0) {
					throw std::runtime_error("embed: " + last_err);
				}
				int backoff_ms = 400 * (1 << std::min(attempt++, 5));
				std::this_thread::sleep_for(std::chrono::milliseconds(std::min(backoff_ms, 8000)));
				continue;
			}
			if (res->status < 200 || res->status >= 300) {
				std::string msg;
				FindStringValue(res->body, "message", msg);
				throw std::runtime_error("embed: provider returned HTTP " + std::to_string(res->status) +
				                         (msg.empty() ? "" : ": " + msg));
			}
			break;
		}

		yj::yyjson_doc *doc = yj::yyjson_read(res->body.c_str(), res->body.size(), 0);
		yj::yyjson_val *root = doc ? yj::yyjson_doc_get_root(doc) : nullptr;
		yj::yyjson_val *data = root ? yj::yyjson_obj_get(root, "data") : nullptr;
		if (!data || !yj::yyjson_is_arr(data)) {
			if (doc) {
				yj::yyjson_doc_free(doc);
			}
			throw std::runtime_error("embed: no data[] in response: " + res->body.substr(0, 300));
		}
		std::vector<std::vector<float>> chunk(yj::yyjson_arr_size(data));
		yj::yyjson_arr_iter it;
		yj::yyjson_arr_iter_init(data, &it);
		yj::yyjson_val *el;
		size_t seq = 0;
		while ((el = yj::yyjson_arr_iter_next(&it))) {
			long idx = (long)seq;
			yj::yyjson_val *iv = yj::yyjson_obj_get(el, "index");
			if (iv && yj::yyjson_is_int(iv)) {
				idx = (long)yj::yyjson_get_sint(iv);
			}
			std::vector<float> v;
			yj::yyjson_val *emb = yj::yyjson_obj_get(el, "embedding");
			if (emb && yj::yyjson_is_arr(emb)) {
				v.reserve(yj::yyjson_arr_size(emb));
				yj::yyjson_arr_iter eit;
				yj::yyjson_arr_iter_init(emb, &eit);
				yj::yyjson_val *f;
				while ((f = yj::yyjson_arr_iter_next(&eit))) {
					double num = yj::yyjson_is_real(f)  ? yj::yyjson_get_real(f)
					             : yj::yyjson_is_int(f) ? (double)yj::yyjson_get_sint(f)
					                                    : (double)yj::yyjson_get_uint(f);
					v.push_back((float)num);
				}
			}
			if (idx >= 0 && (size_t)idx < chunk.size()) {
				chunk[(size_t)idx] = std::move(v);
			} else if (seq < chunk.size()) {
				chunk[seq] = std::move(v);
			}
			seq++;
		}
		yj::yyjson_doc_free(doc);
		for (auto &v : chunk) {
			out.push_back(std::move(v));
		}
	}
	uint64_t wall_ms =
	    (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
	RecordEmbed(texts.size(), wall_ms);
	return out;
}

} // namespace duckthink

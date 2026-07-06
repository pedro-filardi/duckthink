//===----------------------------------------------------------------------===//
// provider.hpp — provider-agnostic LLM plumbing for the semantic_join extension.
//
// Deliberately free of any DuckDB headers so the httplib include in llm.cpp
// stays isolated from the DuckDB translation units. The DuckDB side resolves a
// model + secret into an LlmCall and hands it here; this layer only knows how
// to talk HTTP to an OpenAI-compatible chat endpoint (which Ollama also serves
// at /v1/chat/completions), so adding a provider is mostly a base_url + auth.
//===----------------------------------------------------------------------===//
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace duckthink {

// A model declared via CREATE MODEL('name', 'model-id', 'provider').
struct LlmModel {
	std::string name;     // user-facing handle, e.g. 'summarizer'
	std::string model_id; // provider model, e.g. 'gpt-4o' or 'llama3.2'
	std::string provider; // 'openai', 'ollama', ...
};

// A fully-resolved call: model_id + the endpoint/credentials pulled from the
// matching DuckDB secret. Everything the transport needs, nothing DuckDB.
struct LlmCall {
	std::string provider;      // for dispatch / error messages
	std::string model_id;      // provider model id
	std::string base_url;      // e.g. http://localhost:11434 or https://api.openai.com
	std::string api_key;       // empty for keyless local providers
	double temperature = 0.0;  // deterministic by default
	int timeout_seconds = 120; // per-request wall clock
	int max_retries = 4;       // retries on 5xx / connection errors (exp backoff)
	// A 429 (rate limit) is not a failure, just "wait": we keep re-trying it,
	// honoring the server's Retry-After, for up to this many seconds total per
	// call (separate from max_retries). 0 disables the special handling.
	int rate_limit_wait_seconds = 300;
	int max_tokens = 2048;       // response cap; required by Anthropic, ignored by OpenAI/Ollama
	std::string response_format; // raw JSON for "response_format" (empty => none); e.g. {"type":"json_object"}
	bool log = false;            // emit a one-line stderr trace per call (SET duckthink_log)
};

// --- Model registry (in-memory, process-wide, thread-safe) -----------------
// Declaring a model twice replaces the earlier definition (CREATE OR REPLACE
// semantics). Names are matched case-insensitively.
void RegisterModel(const LlmModel &model);
bool LookupModel(const std::string &name, LlmModel &out);
bool DropModel(const std::string &name);
std::vector<LlmModel> AllModels();

// A named, reusable prompt template declared via CREATE PROMPT('name', 'text').
struct LlmPrompt {
	std::string name; // handle, e.g. 'summarize'
	std::string body; // the template / instruction text
};

// --- Prompt registry (same lifetime/threading semantics as models) ----------
void RegisterPrompt(const LlmPrompt &prompt);
bool LookupPrompt(const std::string &name, LlmPrompt &out);
bool DropPrompt(const std::string &name);
std::vector<LlmPrompt> AllPrompts();

// --- Provider dispatch ------------------------------------------------------
// Runs one chat completion and returns the assistant text. Throws
// std::runtime_error with a human-readable message on any transport/HTTP/JSON
// or provider error (unreachable host, non-2xx, error body, missing content).
std::string LlmChat(const LlmCall &call, const std::string &system_prompt, const std::string &user_prompt);

// Embed a batch of texts via the provider's OpenAI-compatible /v1/embeddings endpoint
// (OpenAI, Ollama). Returns one vector per input, in input order. Throws for providers
// without an embeddings endpoint (Anthropic) or on any transport/HTTP error.
std::vector<std::vector<float>> LlmEmbed(const LlmCall &call, const std::vector<std::string> &texts);

// Default base_url for a provider when the secret doesn't pin one.
std::string DefaultBaseUrl(const std::string &provider);

// Download `url` to `dest_path`, following redirects (HuggingFace resolve/ ->
// CDN) and streaming to disk (models are 100+ MB). Writes to a .part file and
// renames on success. Returns false with a message in `err` on any failure.
// https targets require a TLS build (OpenSSL); http works without.
bool DownloadToFile(const std::string &url, const std::string &dest_path, std::string &err);

// Lenient extraction of a JSON boolean / integer value for `key` out of a
// (possibly chatty) model response. Return false if not present/parseable.
bool JsonFindBool(const std::string &body, const std::string &key, bool &out);
bool JsonFindInt(const std::string &body, const std::string &key, long &out);

// --- Observability ----------------------------------------------------------
// Process-wide counters since load (or the last ResetStats). Deliberately about
// EFFICIENCY, not just totals: comparing api_calls to the rows a caller handled
// shows batching's effect, cache_hits shows dedup's, and embed rows carry zero
// tokens so the "local is free, only LLM calls cost" story is explicit. All
// thread-safe; LlmChat records every real call at its single choke point.
struct LlmStat {
	std::string model; // "provider:model_id"
	uint64_t api_calls = 0;
	uint64_t cache_hits = 0;
	uint64_t input_tokens = 0;
	uint64_t output_tokens = 0;
	uint64_t retries = 0;         // transient (5xx/conn) retry attempts
	uint64_t rate_limited_ms = 0; // total time slept waiting out 429s
	uint64_t wall_ms = 0;         // total time inside HTTP calls
	uint64_t errors = 0;          // calls that ultimately failed
};
struct EmbedStat {
	uint64_t texts = 0; // texts embedded locally (no API, no tokens)
	uint64_t wall_ms = 0;
};

void RecordLlmCall(const std::string &provider, const std::string &model_id, uint64_t input_tokens,
                   uint64_t output_tokens, uint64_t retries, uint64_t rate_limited_ms, uint64_t wall_ms, bool error);
void RecordCacheHit(const std::string &provider, const std::string &model_id);
void RecordEmbed(uint64_t texts, uint64_t wall_ms);
std::vector<LlmStat> SnapshotLlmStats();
EmbedStat SnapshotEmbedStats();
void ResetStats();

} // namespace duckthink

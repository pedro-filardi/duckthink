# duckthink — `ASK()` natural-language SQL for DuckDB, grounded in your dbt Semantic Layer

A DuckDB C++ extension with a single focus: **`ASK()`** — ask a question in plain
language, get back a **typed, executed** DuckDB result. It generates one read-only
`SELECT`, runs it, and streams the rows. It works over the plain DuckDB catalog out of the
box; point it at your **dbt Semantic Layer** and the generated SQL uses your real entities
(join keys), dimensions and metric definitions instead of guessing.

Works with **OpenAI, Ollama, or Anthropic**. Built and tested against **DuckDB v1.5.4**.

```sql
ASK('average resolution time by team, slowest first')
  ON (tickets, agents)
  RETURN (team VARCHAR, avg_hours DECIMAL);
```
```
┌──────────┬───────────┐
│ team     │ avg_hours │
├──────────┼───────────┤
│ billing  │     9.100 │
│ platform │     2.400 │
└──────────┴───────────┘
```

| function | what it does |
|---|---|
| `ASK('q') ON (tables) RETURN (schema)` | natural language → **one read-only SELECT**, executed and typed |
| `ask_sql('q', 'tables', 'schema')` | the same, but returns the **generated SQL** as text (no execution) |
| `CREATE MODEL` / `DROP MODEL` | register an LLM the way you register a source |
| `CREATE SECRET (TYPE openai\|ollama\|anthropic, ...)` | provider credentials (the key never leaves the secret) |
| `duckthink_metrics()` | per-model API calls, cache hits, tokens & latency — see what you spend |

---

## Quickstart

```sql
LOAD 'duckthink';

-- 1. Credentials (the API key lives only in the secret, never in a query string).
CREATE SECRET oai (TYPE openai, API_KEY 'sk-...');

-- 2. Register a model and make it the default ASK uses.
CREATE MODEL('gpt', 'gpt-4o-mini', 'openai');
SET ask_model = 'gpt';

-- 3. Ask.
ASK('top 5 agents by tickets resolved this month')
  ON (tickets, agents)
  RETURN (agent VARCHAR, resolved BIGINT);
```

`ASK` writes exactly one `SELECT` over the tables you list in `ON (...)`, verifies it is
read-only (via DuckDB's own parser, not a string check), executes it, and streams the
result with the column names/types you declared in `RETURN (...)`. If the query errors,
`ASK` feeds the error back to the model and retries (see `ask_max_retries`).

Prefer to inspect the SQL first? Use `ask_sql(...)`, which returns the query as text:

```sql
SELECT sql FROM ask_sql('top 5 agents by tickets resolved this month',
                        'tickets, agents', 'agent VARCHAR, resolved BIGINT');
```

### Local / other providers

```sql
CREATE SECRET local (TYPE ollama, ENDPOINT 'http://localhost:11434');
CREATE MODEL('llama', 'llama3.2', 'ollama');
SET ask_model = 'llama';

CREATE SECRET anthropic (TYPE anthropic, API_KEY 'sk-ant-...');
CREATE MODEL('claude', 'claude-3-5-haiku', 'anthropic');
```

---

## Grounding ASK in your dbt Semantic Layer

Point `ASK` at a dbt manifest and it stops guessing. Set it once:

```sql
SET ask_dbt_semantic_manifest = '/path/to/dbt/target/manifest.json';
-- (a MetricFlow target/semantic_manifest.json works too)
```

For every table in the `ON (...)` scope, `ASK` reads and injects into the prompt:

- **Semantic models** — each model's **entities** (the real join keys), **dimensions**
  (what you group by), and **measures** (what you aggregate, with their aggregation).
- **Metrics** — MetricFlow metric definitions (`simple` / `ratio` / `derived`), so
  "resolution time" resolves to `SUM(resolution_min) / 60` and "SLA breach rate" to the right ratio.
- **Column & table descriptions + synonyms** — from your `schema.yml`.
- **Relationships** — join keys inferred from `relationships` generic tests.

The effect, from the example above — the model was told `tickets.agent` is a foreign entity
(`agent_id`), `team` is a dimension on `agents`, and the `resolution_hours` metric averages
`resolution_min / 60` per ticket. It produced:

```sql
SELECT a.team, SUM(t.resolution_min) / 60.0 / COUNT(t.ticket_id) AS avg_hours
FROM agents a JOIN tickets t ON a.agent_id = t.agent_id
GROUP BY a.team
```

The join key came from the **entity**, the grouping from the **dimension**, and the
aggregation from the **measure/metric** — none of it guessed from column names.

### Scope to a business domain with dbt tags

Often you want to bound a question to a **context scope** — a **business domain** the answer
should stay within — rather than a hand-picked table list. Point `ASK` at a set of dbt
**tags** and it resolves that scope from the manifest; the `ON (...)` clause becomes optional:

```sql
SET ask_dbt_scope_tags = 'taxi';           -- every model/source tagged `taxi`
ASK('overall utilization') RETURN (utilization DECIMAL);   -- no ON needed
```

`ASK` pulls in exactly the tables carrying that tag and **only** their semantic models and
metrics. A `taxi`-scoped question never sees `finance` tables or metrics — the business
domains are isolated, deterministically, by the tags your team already maintains. Explicit
`ON (...)` tables are unioned in when you want both.

**Two-stage retrieval.** A context scope can span many tables. When it resolves to more than
`ask_scope_max_tables` (default 12) tables, `ASK` narrows it to the relevant ones before
building the real prompt — so a 1,000-table domain still sends a handful of tables to the
generation step:

1. **Vector recall** *(optional, `ask_embed_model`)* — every candidate table's summary is
   embedded **once and cached** (keyed by manifest mtime); each question is embedded and the
   candidates are ranked by cosine, keeping the top `ask_retrieve_k` (default 30). Cheap and
   scales to thousands of tables. Uses the provider's embeddings API (OpenAI/Ollama) — no
   local model, no index to maintain.
2. **LLM rerank** — the model picks the minimal relevant set from the survivors (names +
   descriptions), capped at `ask_scope_max_tables`.

Without `ask_embed_model` you get stage 2 only (fine up to a few dozen tables). With it, the
embedding stage does the coarse filtering so the rerank — and the final prompt — stay small.
Both stages are memoized, so repeated questions in a session cost nothing extra.

```sql
CREATE MODEL('emb', 'text-embedding-3-small', 'openai');
SET ask_embed_model = 'emb';   -- turns on vector recall
```

### No dbt? ASK over the plain catalog

dbt is optional. `ASK` finds its tables in one of three ways:

1. the tables you name in **`ON (...)`**, or
2. the tables a **`ask_dbt_scope_tags`** value resolves to (needs a manifest), or
3. with neither — **every table in the DuckDB catalog**, enriched with each table's column
   names, narrowed by the same two-stage retrieval (turn on `ask_embed_model` for large
   databases).

So on a plain DuckDB with no dbt at all you can still just ask:

```sql
SET ask_embed_model = 'emb';        -- optional; recommended for large catalogs
ASK('tickets opened per month, most recent first') RETURN (month VARCHAR, tickets BIGINT);
```

Without the semantic layer you lose the metric definitions, so *non-obvious* business
metrics degrade to the model's best guess — but everyday questions work out of the box, and
adding a manifest later is what turns "close enough" into "exactly your metric".

---

## Settings

| setting | default | meaning |
|---|---|---|
| `ask_model` | — | registered model `ASK` uses (declare it with `CREATE MODEL`) |
| `ask_dbt_semantic_manifest` | `''` | path to a dbt `manifest.json` / `semantic_manifest.json`; empty = off |
| `ask_dbt_scope_tags` | `''` | comma-separated dbt tags: scope `ASK` to a **domain** instead of naming tables (see below) |
| `ask_scope_max_tables` | `12` | above this many candidate tables, `ASK` narrows to the relevant ones before the prompt; `0` disables it |
| `ask_embed_model` | `''` | registered embedding model (openai/ollama): adds a cheap **vector recall** stage before the LLM rerank |
| `ask_retrieve_k` | `30` | how many tables the embedding recall keeps before the rerank (only when `ask_embed_model` is set) |
| `ask_max_retries` | `2` | times a failing generated query's error is fed back for self-correction (0 = one shot) |
| `llm_max_tokens` | `2048` | max response tokens per call (required by Anthropic) |
| `llm_rate_limit_wait` | `300` | seconds to wait out HTTP 429s (honoring `Retry-After`) before failing |
| `duckthink_log` | `false` | one-line stderr trace per LLM call (provider:model, tokens, ms) |

Identical requests are memoized (temperature 0), so re-running the same `ASK` in a session
costs zero API calls — see `duckthink_metrics()` for calls vs cache hits.

---

## Build

```bash
git clone --recurse-submodules <repo> && cd duckthink
GEN=ninja make            # builds DuckDB + the extension
```

Pure C++: DuckDB's bundled httplib for transport, OpenSSL for `https://` (found
automatically via Homebrew on macOS; without it, `http` providers like Ollama still work,
or use the bundled `openai_proxy`). No Python, no local inference, no heavyweight deps.

The loadable extension lands at
`build/release/extension/duckthink/duckthink.duckdb_extension`.

---

## Safety

- **Read-only.** The generated statement is parsed and rejected unless it is a single
  read-only `SELECT` — enforced through DuckDB's parser, not a string prefix check.
- **Scoped.** `ASK` sees only the tables in scope — the ones you name in `ON (...)`, the
  ones a dbt tag resolves to, or (with neither) the tables retrieval selects from the catalog.
- **Keys stay in secrets.** The API key is read from the DuckDB secret at call time and is
  never part of a query string, a log line, or the generated SQL.

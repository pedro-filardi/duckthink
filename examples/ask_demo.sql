-- duckthink ASK() demo. Run with the extension loaded, e.g.:
--   build/release/duckdb mydb.duckdb -unsigned -c ".read examples/ask_demo.sql"
-- (the OPENAI_API_KEY goes into a secret; it never appears in a query or a log line).

-- 1. Credentials + a model, and make it the default ASK uses.
CREATE SECRET oai (TYPE openai, API_KEY 'sk-...');   -- or TYPE ollama / anthropic
CREATE MODEL('gpt', 'gpt-4o-mini', 'openai');
SET ask_model = 'gpt';

-- Some tables to ask about.
CREATE TABLE trips   (trip_id INT, fare DECIMAL(10,2), tips DECIMAL(10,2), duration_min INT, occupied_min INT);
CREATE TABLE drivers (driver_id INT, name VARCHAR, vehicle_type VARCHAR);

-- 2. ASK over named tables → one read-only SELECT, executed and typed.
ASK('total fare revenue and average tip rate')
  ON (trips)
  RETURN (revenue DECIMAL, tip_rate DECIMAL);

-- ask_sql(...) returns the generated SQL instead of running it.
SELECT sql FROM ask_sql('fleet utilization: occupied over total minutes', 'trips', 'x DECIMAL');

-- 3. No ON clause, no dbt: ASK ranges over the whole DuckDB catalog and picks the
--    relevant table(s) itself (turn on embedding retrieval for large schemas).
CREATE MODEL('emb', 'text-embedding-3-small', 'openai');
SET ask_embed_model = 'emb';
ASK('which vehicle types drive the most revenue') RETURN (vehicle_type VARCHAR, revenue DECIMAL);

-- 4. Ground generation in a dbt Semantic Layer: metrics resolve to their real formulas,
--    entities become join keys, and tags scope ASK to a domain.
--   SET ask_dbt_semantic_manifest = '/path/to/dbt/target/manifest.json';
--   SET ask_dbt_scope_tags = 'taxi';
--   ASK('overall fleet utilization') RETURN (utilization DECIMAL);   -- no ON needed

-- What each LLM call cost (API calls, cache hits, tokens, latency):
SELECT * FROM duckthink_metrics();

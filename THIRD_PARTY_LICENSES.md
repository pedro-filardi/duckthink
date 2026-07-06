# Third-party components

duckthink links the following components at build or run time. Their licenses apply to
the corresponding parts of any distributed binary.

| Component | Used for | License | Notes |
|---|---|---|---|
| [DuckDB](https://github.com/duckdb/duckdb) | host database / extension API | MIT | build dependency (submodule) |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | HTTP transport to LLM providers | MIT | DuckDB's bundled copy |
| [yyjson](https://github.com/ibireme/yyjson) | JSON parsing of LLM responses | MIT | DuckDB's bundled copy |
| [OpenSSL](https://www.openssl.org/) | TLS for `https://` provider endpoints | Apache-2.0 | linked via vcpkg; optional (http providers work without it) |

duckthink calls LLM/embedding APIs at query time only; it does not bundle or redistribute
any model weights.

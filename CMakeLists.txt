cmake_minimum_required(VERSION 3.5)

# Set extension name here
set(TARGET_NAME duckthink)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

# Build as C++11 — the DuckDB extension default. The community-extensions toolchain
# appends -std=c++11 last (it wins over any -std=c++17), so the sources are kept
# C++11-clean rather than relying on a fragile flag-ordering override.
set(CMAKE_CXX_STANDARD "11" CACHE STRING "C++ standard to enforce")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# duckthink is an ASK-only, cloud-model extension: no local inference, no
# llama.cpp/GGML, no embeddings. It just needs DuckDB's bundled httplib to talk
# to an OpenAI-compatible chat endpoint (and OpenSSL, below, for https).
include_directories(
  src/include
  ${CMAKE_CURRENT_LIST_DIR}/duckdb/third_party/httplib)

set(EXTENSION_SOURCES
  src/duckthink_extension.cpp   # keyword parser (ASK / MODEL DDL) + registration
  src/provider.cpp              # LLM transport + model registry (no DuckDB deps)
  src/ai_functions.cpp)         # provider secrets, model registry, ASK text-to-SQL

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# --- Optional TLS: link OpenSSL so https:// providers (OpenAI) work directly.
# When OpenSSL is not available the extension still builds; https endpoints then
# require a local HTTP proxy (see README) and http providers (Ollama) work as-is.
# Homebrew's OpenSSL isn't on CMake's default search path.
if(APPLE AND NOT DEFINED OPENSSL_ROOT_DIR)
  execute_process(COMMAND brew --prefix openssl@3
                  OUTPUT_VARIABLE _brew_ssl OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(_brew_ssl)
    set(OPENSSL_ROOT_DIR ${_brew_ssl})
  endif()
endif()
# Link OpenSSL statically so the loadable extension carries no absolute path to a
# system libssl — portable to other machines. Override with -DOPENSSL_USE_STATIC_LIBS=OFF.
if(NOT DEFINED OPENSSL_USE_STATIC_LIBS)
  set(OPENSSL_USE_STATIC_LIBS ON)
endif()
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
  # Prefer the static archives (in the same dir as the found lib) so no absolute
  # libssl.dylib path is baked into the loadable extension — portable across machines.
  get_filename_component(_ssl_dir "${OPENSSL_SSL_LIBRARY}" DIRECTORY)
  if(EXISTS "${_ssl_dir}/libssl.a" AND EXISTS "${_ssl_dir}/libcrypto.a")
    set(DUCKTHINK_SSL_LIBS "${_ssl_dir}/libssl.a" "${_ssl_dir}/libcrypto.a")
    message(STATUS "duckthink: OpenSSL found (static) — TLS on, portable binary")
  else()
    set(DUCKTHINK_SSL_LIBS OpenSSL::SSL OpenSSL::Crypto)
    message(STATUS "duckthink: OpenSSL found (shared) — TLS on, binary depends on system libssl")
  endif()
else()
  message(WARNING "duckthink: OpenSSL not found — building WITHOUT TLS (https via proxy only)")
endif()

foreach(_t ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
  if(OpenSSL_FOUND)
    target_compile_definitions(${_t} PRIVATE DUCKTHINK_TLS)
    target_include_directories(${_t} PRIVATE ${OPENSSL_INCLUDE_DIR})
    target_link_libraries(${_t} ${DUCKTHINK_SSL_LIBS})
  endif()
endforeach()

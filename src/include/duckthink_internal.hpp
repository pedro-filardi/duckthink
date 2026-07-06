//===----------------------------------------------------------------------===//
// duckthink_internal.hpp — symbols shared between the extension's translation
// units. duckthink_extension.cpp holds the keyword parser + registration entry;
// ai_functions.cpp registers the provider secrets, model registry and the ASK
// text-to-SQL functions.
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb.hpp"

namespace duckdb {

class ExtensionLoader;

// Registers provider secrets, the model registry, call metrics, and the ASK /
// ask_sql functions. Defined in ai_functions.cpp, called from LoadInternal.
void RegisterAiFunctions(ExtensionLoader &loader);

} // namespace duckdb

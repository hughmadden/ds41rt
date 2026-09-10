#include "ds41rt_native.h"

#include <cassert>
#include <cstdint>
#include <iostream>

#ifndef DS41RT_XGRAMMAR_TEST_TOKENIZER
#error "DS41RT_XGRAMMAR_TEST_TOKENIZER is required"
#endif

namespace {

bool Allows(const uint32_t* mask, uint32_t token) {
  return (mask[token / 32] & (uint32_t{1} << (token % 32))) != 0;
}

}  // namespace

int main() {
  char error[2048] = {};
  const int32_t stops[] = {6};
  void* compiler = nullptr;
  assert(ds41rt_xgrammar_compiler_create(
             DS41RT_XGRAMMAR_TEST_TOKENIZER, 8, stops, 1, &compiler, error, sizeof(error)) ==
         DS41RT_STATUS_OK);
  assert(compiler != nullptr);

  void* grammar = nullptr;
  assert(ds41rt_xgrammar_compile(
             compiler, DS41RT_XGRAMMAR_JSON_SCHEMA,
             R"({"type":"object","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false})",
             1, &grammar, error, sizeof(error)) == DS41RT_STATUS_OK);
  assert(grammar != nullptr);

  void* matcher = nullptr;
  assert(ds41rt_xgrammar_matcher_create(grammar, &matcher, error, sizeof(error)) ==
         DS41RT_STATUS_OK);
  uint32_t mask[1] = {};
  int needs_mask = 0;
  assert(ds41rt_xgrammar_matcher_fill_bitmask(
             matcher, mask, 1, &needs_mask, error, sizeof(error)) == DS41RT_STATUS_OK);
  assert(needs_mask != 0);
  assert(Allows(mask, 1));
  assert(!Allows(mask, 4));

  int accepted = 0;
  assert(ds41rt_xgrammar_matcher_accept_token(
             matcher, 1, &accepted, error, sizeof(error)) == DS41RT_STATUS_OK);
  assert(accepted != 0);
  for (int whitespace = 0; whitespace < 16; ++whitespace) {
    accepted = 0;
    assert(ds41rt_xgrammar_matcher_accept_token(
               matcher, 7, &accepted, error, sizeof(error)) == DS41RT_STATUS_OK);
    assert(accepted != 0);
  }
  mask[0] = 0;
  assert(ds41rt_xgrammar_matcher_fill_bitmask(
             matcher, mask, 1, &needs_mask, error, sizeof(error)) == DS41RT_STATUS_OK);
  assert(!Allows(mask, 7));
  assert(Allows(mask, 2));

  assert(ds41rt_xgrammar_matcher_destroy(matcher) == DS41RT_STATUS_OK);
  matcher = nullptr;
  assert(ds41rt_xgrammar_matcher_create(grammar, &matcher, error, sizeof(error)) ==
         DS41RT_STATUS_OK);

  for (const uint32_t token : {1U, 2U, 3U, 4U, 5U}) {
    accepted = 0;
    assert(ds41rt_xgrammar_matcher_accept_token(
               matcher, token, &accepted, error, sizeof(error)) == DS41RT_STATUS_OK);
    assert(accepted != 0);
  }
  int completed = 0;
  assert(ds41rt_xgrammar_matcher_is_completed(
             matcher, &completed, error, sizeof(error)) == DS41RT_STATUS_OK);
  assert(completed != 0);
  mask[0] = 0;
  assert(ds41rt_xgrammar_matcher_fill_bitmask(
             matcher, mask, 1, &needs_mask, error, sizeof(error)) == DS41RT_STATUS_OK);
  assert(Allows(mask, 6));

  void* forked = nullptr;
  assert(ds41rt_xgrammar_matcher_fork(matcher, &forked, error, sizeof(error)) ==
         DS41RT_STATUS_OK);
  assert(ds41rt_xgrammar_matcher_destroy(forked) == DS41RT_STATUS_OK);
  assert(ds41rt_xgrammar_matcher_destroy(matcher) == DS41RT_STATUS_OK);
  assert(ds41rt_xgrammar_grammar_destroy(grammar) == DS41RT_STATUS_OK);
  assert(ds41rt_xgrammar_compiler_destroy(compiler) == DS41RT_STATUS_OK);
  std::cout << "ds41rt_xgrammar_selftest: ok\n";
  return 0;
}

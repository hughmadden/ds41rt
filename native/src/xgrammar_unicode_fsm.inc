// Included in the generated pinned grammar_functor.cc translation unit.
// Negation applies to Unicode codepoints, before conversion to UTF-8 bytes.
FSMWithStartEnd GrammarFSMBuilderImpl::BuildNegativeCharacterClass(const GrammarExpr& expr) {
  std::vector<std::pair<uint32_t, uint32_t>> excluded;
  for (int i = 1; i < expr.size(); i += 2)
    excluded.emplace_back(static_cast<uint32_t>(expr[i]), static_cast<uint32_t>(expr[i + 1]));
  // UTF-8 has no encodings for surrogate codepoints.
  excluded.emplace_back(0xD800, 0xDFFF);
  std::sort(excluded.begin(), excluded.end());

  FSMWithStartEnd result;
  int start = result.AddState();
  int end = expr.type == ExprType::kCharacterClassStar ? start : result.AddState();
  result.SetStartState(start);
  result.AddEndState(end);
  uint32_t next = 0;
  for (const auto& range : excluded) {
    auto lo = std::min(range.first, uint32_t{0x110000});
    auto hi = std::min(range.second, uint32_t{0x10FFFF});
    if (next < lo)
      AddCharacterRange(result, start, end, CodepointToPackedUTF8(next), CodepointToPackedUTF8(lo - 1));
    next = std::max(next, hi + 1);
  }
  if (next <= 0x10FFFF)
    AddCharacterRange(result, start, end, CodepointToPackedUTF8(next), CodepointToPackedUTF8(0x10FFFF));
  return result;
}

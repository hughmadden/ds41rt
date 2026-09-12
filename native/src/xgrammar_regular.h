#pragma once
#include "ebnf_script_creator.h"
#include "fsm.h"
#include <xgrammar/grammar.h>

namespace xgrammar {
// Bounded operations for regular schema fragments. Recursive JSON grammars
// with non-tail recursion are deliberately not treated as regular languages.
FSMWithStartEnd V41RegularFSM(const Grammar& grammar);
FSMWithStartEnd V41Intersect(const FSMWithStartEnd& lhs, const FSMWithStartEnd& rhs);
FSMWithStartEnd V41Subtract(const FSMWithStartEnd& lhs, const FSMWithStartEnd& rhs);
bool V41Empty(const FSMWithStartEnd& fsm);
std::string V41EmitFSM(const FSMWithStartEnd& fsm, EBNFScriptCreator& script);
}  // namespace xgrammar

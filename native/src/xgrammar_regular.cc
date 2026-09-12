#include "xgrammar_regular.h"
#include "grammar_functor.h"
#include "grammar_printer.h"
#include <algorithm>
#include <functional>
#include <map>
#include <queue>
#include <stdexcept>

namespace xgrammar {
namespace {
constexpr int kStateLimit = 16384;
FSMWithStartEnd Unwrap(Result<FSMWithStartEnd> result) {
  if (result.IsErr()) throw std::move(result).UnwrapErr();
  return std::move(result).Unwrap();
}
class RegularBuilder {
 public:
  explicit RegularBuilder(const Grammar& grammar) : grammar_(grammar) {}
  FSMWithStartEnd Build() {
    int from = State(), to = State();
    Rule(grammar_->GetRootRuleId(), from, to);
    std::vector<bool> ends(fsm_.NumStates(), false);
    ends[to] = true;
    return Unwrap(FSMWithStartEnd(fsm_, from, ends).ToDFA(kStateLimit));
  }
 private:
  using Type = Grammar::Impl::GrammarExprType;
  Grammar grammar_;
  FSM fsm_;
  std::map<int, std::pair<int, int>> active_;
  int State() {
    if (fsm_.NumStates() >= kStateLimit * 4)
      throw std::invalid_argument("regular schema fragment exceeds the NFA state budget");
    return fsm_.AddState();
  }
  void Leaf(const FSMWithStartEnd& leaf, int from, int to) {
    if (fsm_.NumStates() + leaf.NumStates() > kStateLimit * 4)
      throw std::invalid_argument("regular schema fragment exceeds the NFA state budget");
    std::vector<int> map;
    fsm_.AddFSM(leaf.GetFsm(), &map);
    fsm_.AddEpsilonEdge(from, map[leaf.GetStart()]);
    for (int i = 0; i < leaf.NumStates(); ++i)
      if (leaf.IsEndState(i)) fsm_.AddEpsilonEdge(map[i], to);
  }
  void Rule(int id, int from, int to) {
    if (auto it = active_.find(id); it != active_.end()) {
      if (to != it->second.second)
        throw std::invalid_argument("schema intersection contains non-regular recursion");
      fsm_.AddEpsilonEdge(from, it->second.first);
      return;
    }
    const auto& rule = grammar_->GetRule(id);
    if (rule.lookahead_assertion_id != -1)
      throw std::invalid_argument("regular schema fragment contains a lookahead assertion");
    active_[id] = {from, to};
    Expr(rule.body_expr_id, from, to);
    active_.erase(id);
  }
  void Expr(int id, int from, int to) {
    auto expr = grammar_->GetGrammarExpr(id);
    switch (expr.type) {
      case Type::kEmptyStr: fsm_.AddEpsilonEdge(from, to); break;
      case Type::kByteString: Leaf(GrammarFSMBuilder::ByteString(expr), from, to); break;
      case Type::kCharacterClass:
      case Type::kCharacterClassStar: Leaf(GrammarFSMBuilder::CharacterClass(expr), from, to); break;
      case Type::kRuleRef: Rule(expr[0], from, to); break;
      case Type::kChoices:
        for (int child : expr) Expr(child, from, to);
        break;
      case Type::kSequence:
        if (expr.size() == 0) fsm_.AddEpsilonEdge(from, to);
        for (int i = 0; i < expr.size(); ++i) {
          int next = i + 1 == expr.size() ? to : State();
          Expr(expr[i], from, next);
          from = next;
        }
        break;
      case Type::kRepeat: {
        int lower = expr[1], upper = expr[2];
        if (lower > kStateLimit || upper > kStateLimit)
          throw std::invalid_argument("regular schema repetition exceeds the state budget");
        for (int i = 0; i < lower; ++i) {
          int next = State(); Rule(expr[0], from, next); from = next;
        }
        fsm_.AddEpsilonEdge(from, to);
        if (upper == -1) Rule(expr[0], from, from);
        else for (int i = lower; i < upper; ++i) {
          int next = State(); Rule(expr[0], from, next); from = next;
          fsm_.AddEpsilonEdge(from, to);
        }
        break;
      }
      default: throw std::invalid_argument("schema fragment is not a regular character grammar");
    }
  }
};

// Convert byte-DFA transitions back to whole Unicode scalar transitions.
// EBNF string escapes encode codepoints, so printing individual UTF-8 bytes
// would corrupt the language. Intermediate UTF-8 states are not emitted.
class UnicodeEmitter {
 public:
  UnicodeEmitter(FSMWithStartEnd fsm, EBNFScriptCreator& script)
      : fsm_(std::move(fsm)), script_(script), names_(fsm_.NumStates()), alive_(fsm_.NumStates()) {
    std::vector<std::vector<int>> reverse(fsm_.NumStates());
    std::queue<int> queue;
    for (int i = 0; i < fsm_.NumStates(); ++i) {
      for (const auto& edge : fsm_.GetFsm().GetEdges(i)) reverse[edge.target].push_back(i);
      if (fsm_.IsEndState(i)) { alive_[i] = true; queue.push(i); }
    }
    while (!queue.empty()) {
      auto next = queue.front(); queue.pop();
      for (int previous : reverse[next]) if (!alive_[previous]) {
        alive_[previous] = true; queue.push(previous);
      }
    }
  }
  std::string Emit() {
    int root = fsm_.GetStart();
    names_[root] = script_.AllocateRuleName("v41_dfa");
    std::queue<int> queue; queue.push(root);
    while (!queue.empty()) {
      int state = queue.front(); queue.pop();
      std::map<int, std::vector<std::pair<int, int>>> transitions;
      for (const auto& edge : fsm_.GetFsm().GetEdges(state)) {
        if (!alive_[edge.target]) continue;
        if (!edge.IsCharRange()) throw std::invalid_argument("non-byte edge in schema DFA");
        Add(transitions, edge.target, std::max(0, edge.min), std::min(127, edge.max));
        for (int byte = std::max(0xC2, edge.min); byte <= std::min(0xF4, edge.max); ++byte) {
          int length = byte < 0xE0 ? 2 : byte < 0xF0 ? 3 : 4;
          int prefix = byte & (length == 2 ? 0x1F : length == 3 ? 0xF : 7);
          Continue(edge.target, length - 1, prefix, length == 2 ? 0x80 : length == 3 ? 0x800 : 0x10000, transitions);
        }
      }
      std::vector<std::string> alternatives;
      if (fsm_.IsEndState(state)) alternatives.push_back("\"\"");
      for (auto& [target, ranges] : transitions) {
        std::sort(ranges.begin(), ranges.end());
        std::vector<GrammarBuilder::CharacterClassElement> merged;
        for (auto [lo, hi] : ranges) {
          if (!merged.empty() && lo <= merged.back().upper + 1) merged.back().upper = std::max(hi, merged.back().upper);
          else merged.push_back({lo, hi});
        }
        if (names_[target].empty()) {
          names_[target] = script_.AllocateRuleName("v41_dfa"); queue.push(target);
        }
        GrammarBuilder builder;
        auto body = builder.AddCharacterClass(merged);
        builder.AddRule("root", body);
        alternatives.push_back(GrammarPrinter(builder.Get()).PrintGrammarExpr(body) + " " + names_[target]);
      }
      auto body = alternatives.empty() ? "[^\\x00-\\U0010ffff]" : EBNFScriptCreator::Or(alternatives);
      script_.AddRuleWithAllocatedName(names_[state], body);
    }
    return names_[root];
  }
 private:
  using Transitions = std::map<int, std::vector<std::pair<int, int>>>;
  FSMWithStartEnd fsm_;
  EBNFScriptCreator& script_;
  std::vector<std::string> names_;
  std::vector<bool> alive_;
  size_t work_ = 0;
  static void Add(Transitions& transitions, int target, int lo, int hi) {
    hi = std::min(hi, 0x10FFFF);
    if (lo > hi) return;
    if (lo < 0xD800) transitions[target].push_back({lo, std::min(hi, 0xD7FF)});
    if (hi >= 0xE000) transitions[target].push_back({std::max(lo, 0xE000), hi});
  }
  void Continue(int state, int remaining, int prefix, int minimum, Transitions& transitions) {
    if (++work_ > 32000000) throw std::invalid_argument("Unicode schema expansion exceeds the work budget");
    for (const auto& edge : fsm_.GetFsm().GetEdges(state)) {
      if (!alive_[edge.target] || !edge.IsCharRange()) continue;
      int lo = std::max(0x80, edge.min), hi = std::min(0xBF, edge.max);
      if (lo > hi) continue;
      if (remaining == 1) Add(transitions, edge.target, std::max(minimum, (prefix << 6) | (lo & 0x3F)), (prefix << 6) | (hi & 0x3F));
      else for (int byte = lo; byte <= hi; ++byte)
        Continue(edge.target, remaining - 1, (prefix << 6) | (byte & 0x3F), minimum, transitions);
    }
  }
};
}  // namespace

FSMWithStartEnd V41RegularFSM(const Grammar& grammar) { return RegularBuilder(grammar).Build(); }
FSMWithStartEnd V41Intersect(const FSMWithStartEnd& lhs, const FSMWithStartEnd& rhs) {
  return Unwrap(FSMWithStartEnd::Intersect(lhs, rhs, kStateLimit));
}
FSMWithStartEnd V41Subtract(const FSMWithStartEnd& lhs, const FSMWithStartEnd& rhs) {
  return V41Intersect(lhs, Unwrap(rhs.Not(kStateLimit)));
}
bool V41Empty(const FSMWithStartEnd& fsm) {
  std::unordered_set<int> states;
  fsm.GetReachableStates(&states);
  return std::none_of(states.begin(), states.end(), [&](int state) { return fsm.IsEndState(state); });
}
std::string V41EmitFSM(const FSMWithStartEnd& fsm, EBNFScriptCreator& script) {
  return UnicodeEmitter(Unwrap(fsm.MinimizeDFA(kStateLimit)), script).Emit();
}
}  // namespace xgrammar

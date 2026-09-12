#include "xgrammar_v41_tools.h"
#include "xgrammar_regular.h"
#include "regex_converter.h"
#include "grammar_functor.h"
#include "grammar_printer.h"
#include <stdexcept>

namespace xgrammar {
namespace {
class EmptyIntersection : public std::invalid_argument {
 public:
  EmptyIntersection() : std::invalid_argument("schema intersection has no valid value") {}
};
const std::string kEnd = "</｜DSML｜ parameter>";
const std::string kEndPrefix = "</｜DSML｜";
const std::string kRaw = "v41_raw_string";
const std::string kKey = "v41_parameter_key";
std::string Literal(const std::string& text) { return EBNFScriptCreator::Str(text); }
std::string Replace(std::string text, const std::string& from, const std::string& to) {
  size_t at = 0;
  while ((at = text.find(from, at)) != std::string::npos) {
    text.replace(at, from.size(), to); at += to.size();
  }
  return text;
}
std::string Key(const std::string& name) {
  // The recipe ends a parameter-name token at a literal space, so escape
  // spaces inside the JSON-quoted attribute without changing the decoded key.
  return Replace(picojson::value(name).serialize(false), " ", "\\u0020");
}
std::string SafeJSON(const picojson::value& value) {
  // A DSML terminator inside a JSON string must not become a protocol tag.
  return Replace(value.serialize(false), "<", "\\u003c");
}
picojson::value Parse(const std::string& text) {
  picojson::value value;
  auto error = picojson::parse(value, text);
  if (!error.empty()) throw std::invalid_argument(error);
  return value;
}

// Transduce regex terminals to canonical JSON string content. Repetitions still
// count decoded Unicode characters, including characters emitted as escapes.
class EncodedRegex : public GrammarFunctor<std::string, std::string> {
 public:
  EncodedRegex(EBNFScriptCreator& script, bool spaces) : script_(script), spaces_(spaces) {}
  std::string Apply(const Grammar& grammar) override {
    InitGrammar(grammar);
    for (int i = 0; i < grammar->NumRules(); ++i)
      names_.push_back(script_.AllocateRuleName("v41_encoded"));
    for (int i = 0; i < grammar->NumRules(); ++i) {
      const auto& rule = grammar->GetRule(i);
      if (rule.lookahead_assertion_id != -1)
        throw std::invalid_argument("lookahead in encoded string regex is unsupported");
      script_.AddRuleWithAllocatedName(names_[i], VisitExpr(rule.body_expr_id));
    }
    return names_[grammar->GetRootRuleId()];
  }
 protected:
  std::string Encode(const std::string& text) const {
    auto encoded = SafeJSON(picojson::value(text));
    if (spaces_) encoded = Replace(encoded, " ", "\\u0020");
    return Literal(encoded.substr(1, encoded.size() - 2));
  }
  std::string VisitByteString(const GrammarExpr& expr) override {
    std::string text;
    for (auto byte : expr) text.push_back(static_cast<char>(byte));
    return Encode(text);
  }
  std::string VisitEmptyStr(const GrammarExpr&) override { return "\"\""; }
  std::string VisitRuleRef(const GrammarExpr& expr) override { return names_.at(expr[0]); }
  std::string VisitRepeat(const GrammarExpr& expr) override {
    return EBNFScriptCreator::Repeat(names_.at(expr[0]), expr[1], expr[2]);
  }
  std::string VisitSequence(const GrammarExpr& expr) override {
    std::vector<std::string> parts;
    for (auto child : expr) parts.push_back(VisitExpr(child));
    return EBNFScriptCreator::Concat(parts);
  }
  std::string VisitChoices(const GrammarExpr& expr) override {
    std::vector<std::string> parts;
    for (auto child : expr) parts.push_back(VisitExpr(child));
    return EBNFScriptCreator::Or(parts);
  }
  std::string VisitCharacterClass(const GrammarExpr& expr) override {
    std::vector<int> cuts = {0, 0xD800, 0xE000, 0x110000};
    for (int c = 0; c <= 32; ++c) { cuts.push_back(c); cuts.push_back(c + 1); }
    for (int c : {34, 60, 92}) { cuts.push_back(c); cuts.push_back(c + 1); }
    for (int i = 1; i < expr.size(); i += 2) {
      cuts.push_back(expr[i]); cuts.push_back(expr[i + 1] + 1);
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    std::vector<GrammarBuilder::CharacterClassElement> ordinary;
    std::vector<std::string> alternatives;
    for (size_t i = 1; i < cuts.size(); ++i) {
      int lo = cuts[i - 1], hi = cuts[i] - 1;
      if (lo < 0 || lo >= 0x110000 || (lo >= 0xD800 && lo < 0xE000)) continue;
      bool contains = false;
      for (int j = 1; j < expr.size(); j += 2)
        contains |= expr[j] <= lo && lo <= expr[j + 1];
      if (contains == bool(expr[0])) continue;
      if (lo < 32 || (spaces_ && lo == 32) || lo == 34 || lo == 60 || lo == 92)
        alternatives.push_back(Encode(std::string(1, static_cast<char>(lo))));
      else ordinary.push_back({lo, hi});
    }
    if (!ordinary.empty()) {
      GrammarBuilder builder;
      auto body = builder.AddCharacterClass(ordinary);
      builder.AddRule("root", body);
      alternatives.push_back(GrammarPrinter(builder.Get()).PrintGrammarExpr(body));
    }
    if (alternatives.empty()) return "[^\\x00-\\U0010ffff]";
    return EBNFScriptCreator::Or(alternatives);
  }
  std::string VisitCharacterClassStar(const GrammarExpr& expr) override {
    return "(" + VisitCharacterClass(expr) + ")*";
  }
 private:
  EBNFScriptCreator& script_;
  bool spaces_;
  std::vector<std::string> names_;
};
}  // namespace
V41ToolCallingConverter::V41ToolCallingConverter(RefResolver resolver)
    : JSONSchemaConverter(std::nullopt, std::nullopt, true, 16, resolver, false),
      resolver_(std::move(resolver)) {}
std::string V41ToolCallingConverter::ContextKey(const std::string& key) const {
  return (key_context_ ? "key:" : "") + std::to_string(std::min(level_, 2)) + ":" + key;
}
void V41ToolCallingConverter::AddCache(const std::string& key, const std::string& value) {
  if (!key.empty()) cache_[ContextKey(key)] = value;
}
std::optional<std::string> V41ToolCallingConverter::GetCache(const std::string& key) const {
  auto found = cache_.find(ContextKey(key));
  return found == cache_.end() ? std::nullopt : std::optional<std::string>(found->second);
}
void V41ToolCallingConverter::AddBasicRules() {
  auto previous_level = level_;
  auto previous_key_context = key_context_;
  key_context_ = false;
  level_ = 2;
  JSONSchemaConverter::AddBasicRules();
  level_ = previous_level;
  key_context_ = previous_key_context;
  if (regular_context_) {
    auto raw = V41RegularFSM(Grammar::FromRegex("[^]*"));
    auto forbidden = V41RegularFSM(Grammar::FromRegex("[^]*" + kEndPrefix + "[^]*"));
    auto body = V41EmitFSM(V41Subtract(raw, forbidden), ebnf_script_creator_);
    ebnf_script_creator_.AddRule(kRaw, body);
  } else {
    ebnf_script_creator_.AddRule(kRaw,
        "TagDispatch(loop_after_dispatch=false,excludes=(" + Literal(kEndPrefix) + "))");
  }
  // JSON-quoted attribute names, without raw spaces or control characters.
  ebnf_script_creator_.AddRule(kKey, R"ebnf("\"" ([^\x00-\x20"\\] | "\\" basic_escape)* "\"")ebnf");
}
std::string V41ToolCallingConverter::Flag(const std::string& value, bool string) const {
  if (level_ != 1 || key_context_) return value;
  auto body = "(" + value + ")";
  return Literal(string ? "true\">" : "false\">") + " " +
      (string ? body : GetWhitespacePattern() + " " + body + " " + GetWhitespacePattern());
}
#define V41_SCALAR(Name, Spec) \
std::string V41ToolCallingConverter::Generate##Name(const Spec& spec, const std::string& name) { \
  if (key_context_) return "[^\\x00-\\U0010ffff]"; \
  return Flag(JSONSchemaConverter::Generate##Name(spec, name)); \
}
V41_SCALAR(Integer, IntegerSpec)
V41_SCALAR(Number, NumberSpec)
V41_SCALAR(Boolean, BooleanSpec)
V41_SCALAR(Null, NullSpec)
#undef V41_SCALAR
std::string V41ToolCallingConverter::GenerateString(const StringSpec& spec, const std::string& name) {
  if (key_context_) return EncodedString(spec);
  if (level_ != 1) return JSONSchemaConverter::GenerateString(spec, name);
  std::string body = kRaw;
  if (spec.format) {
    if (auto pattern = JSONFormatToRegexPattern(*spec.format)) body = RegexToEBNF(*pattern, false);
  }
  if (spec.pattern) body = RegexToEBNF(*spec.pattern, false);
  else if (!spec.format && (spec.min_length != 0 || spec.max_length != -1)) {
    body = "[^]{" + std::to_string(spec.min_length) + "," +
        (spec.max_length == -1 ? "" : std::to_string(spec.max_length)) + "}";
  }
  if (regular_context_)
    return Flag(body, true) + " | " + Flag(EncodedString(spec));
  return Flag(body, true);
}
std::string V41ToolCallingConverter::EncodedString(const StringSpec& spec) {
  std::string pattern = "[^]*";
  if (spec.format) {
    if (auto value = JSONFormatToRegexPattern(*spec.format)) pattern = *value;
  }
  if (spec.pattern) {
    pattern = *spec.pattern;
    bool start = !pattern.empty() && pattern.front() == '^';
    size_t escapes = 0;
    if (!pattern.empty()) {
      for (size_t i = pattern.size() - 1; i > 0 && pattern[i - 1] == '\\'; --i) ++escapes;
    }
    bool end = !pattern.empty() && pattern.back() == '$' && escapes % 2 == 0;
    if (end) pattern.pop_back();
    if (start) pattern.erase(0, 1);
    pattern = (start ? "" : "[^]*") + std::string("(") + pattern + ")" + (end ? "" : "[^]*");
  }
  else if (!spec.format && (spec.min_length != 0 || spec.max_length != -1))
    pattern = "[^]{" + std::to_string(spec.min_length) + "," +
        (spec.max_length == -1 ? "" : std::to_string(spec.max_length)) + "}";
  auto rule = EncodedRegex(ebnf_script_creator_, key_context_).Apply(Grammar::FromRegex(pattern));
  return Literal("\"") + " " + rule + " " + Literal("\"");
}
std::string V41ToolCallingConverter::GenerateArray(const ArraySpec& spec, const std::string& name) {
  if (key_context_) return "[^\\x00-\\U0010ffff]";
  ++level_;
  auto value = JSONSchemaConverter::GenerateArray(spec, name);
  --level_;
  return Flag(value);
}
std::string V41ToolCallingConverter::GenerateObject(const ObjectSpec& spec, const std::string& name, bool) {
  if (key_context_) return "[^\\x00-\\U0010ffff]";
  const bool root = level_ == 0;
  if (root && !spec.pattern_properties.empty()) {
    return GeneratePatternObject(spec, name);
  }
  if (root && spec.property_names) {
    return GenerateNamedObject(spec, name);
  }
  ++level_;
  auto value = JSONSchemaConverter::GenerateObject(spec, name, !root);
  --level_;
  return Flag(value);
}
std::string V41ToolCallingConverter::GenerateAny(const AnySpec& spec, const std::string& name) {
  if (key_context_) return EncodedString(StringSpec{});
  if (level_ == 0) {
    ObjectSpec object;
    object.allow_additional_properties = true;
    object.additional_properties_schema = SchemaSpec::Make(AnySpec{});
    return GenerateObject(object, name);
  }
  if (level_ == 1) {
    return Flag(kRaw, true) + " | " + Flag(kBasicNumber + " | " + kBasicBoolean + " | " +
        kBasicNull + " | " + kBasicArray + " | " + kBasicObject);
  }
  return JSONSchemaConverter::GenerateAny(spec, name);
}
std::string V41ToolCallingConverter::GenerateConst(const ConstSpec& spec, const std::string&) {
  auto value = Parse(spec.json_value);
  if (key_context_) {
    if (!value.is<std::string>()) return "[^\\x00-\\U0010ffff]";
    return Literal(Replace(SafeJSON(value), " ", "\\u0020"));
  }
  if (level_ == 0) {
    if (!value.is<picojson::object>()) throw std::invalid_argument("tool arguments must be an object");
    std::string result = GetWhitespacePattern();
    ++level_;
    const auto& object = value.get<picojson::object>();
    for (const auto& key : object.ordered_keys()) {
      result += " " + FormatProperty(key, GenerateConst({object.at(key).serialize(false)}, ""), "", 0) + " " + GetWhitespacePattern();
    }
    --level_;
    return result;
  }
  if (level_ == 1 && value.is<std::string>()) {
    const auto& text = value.get<std::string>();
    if (text.find(kEndPrefix) == std::string::npos) return Flag(Literal(text), true);
    // A JSON-encoded string preserves its type while escaping a reserved tag.
    return Flag(Literal(SafeJSON(value)));
  }
  return Flag(Literal(SafeJSON(value)));
}
std::string V41ToolCallingConverter::GenerateNamedObject(const ObjectSpec& spec, const std::string& name) {
  // Compile a vocabulary-free name matcher once to remove forbidden optional
  // fixed keys and diagnose required names that cannot satisfy propertyNames.
  V41ToolCallingConverter names(resolver_);
  names.key_context_ = true;
  names.level_ = 1;
  auto name_grammar = names.Convert(spec.property_names);
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);
  auto compiled = compiler.CompileGrammar(name_grammar);
  std::vector<ObjectSpec::Property> properties;
  for (const auto& property : spec.properties) {
    GrammarMatcher matcher(compiled);
    auto key = Replace(SafeJSON(picojson::value(property.name)), " ", "\\u0020");
    if (matcher.AcceptString(key) && matcher.IsCompleted()) properties.push_back(property);
    else if (spec.required.count(property.name))
      throw std::invalid_argument("required parameter violates propertyNames: " + property.name);
  }

  SchemaSpecPtr additional;
  if (spec.allow_additional_properties) additional = spec.additional_properties_schema;
  else if (spec.allow_unevaluated_properties) additional = spec.unevaluated_properties_schema;
  if (!additional && (spec.allow_additional_properties || spec.allow_unevaluated_properties))
    additional = SchemaSpec::Make(AnySpec{});

  std::vector<std::string> missing;
  for (const auto& required : spec.required) {
    if (std::none_of(properties.begin(), properties.end(), [&](const auto& p) { return p.name == required; }))
      missing.push_back(required);
  }
  std::sort(missing.begin(), missing.end());
  for (const auto& required : missing) {
    GrammarMatcher matcher(compiled);
    auto key = Replace(SafeJSON(picojson::value(required)), " ", "\\u0020");
    if (!additional || !matcher.AcceptString(key) || !matcher.IsCompleted())
      throw std::invalid_argument("required parameter cannot satisfy propertyNames/additionalProperties: " + required);
    properties.push_back({required, additional});
  }
  if (!additional && spec.min_properties > static_cast<int>(properties.size()))
    throw std::invalid_argument("propertyNames leaves too few parameters to satisfy minProperties");

  ++level_;
  indent_manager_.StartIndent();
  std::string other;
  if (additional) {
    auto key = NamePatternExcluding(spec.property_names, properties);
    auto value = CreateRule(additional, name + "_additional");
    other = FormatOtherProperty(key, value, name, "name");
  }
  std::string result;
  if (!properties.empty()) {
    result = GetPartialRuleForProperties(properties, spec.required, additional, name, "name",
        spec.min_properties, spec.max_properties, other);
    if (spec.required.empty() && spec.min_properties == 0)
      result = "(" + result + ") | " + GetWhitespacePattern();
  } else if (additional && spec.max_properties != 0) {
    result = GetWhitespacePattern() + " " + EBNFScriptCreator::Repeat(
        "(" + other + " " + GetWhitespacePattern() + ")", spec.min_properties, spec.max_properties);
  } else {
    if (spec.min_properties > 0 || !spec.required.empty())
      throw std::invalid_argument("propertyNames leaves too few parameters to satisfy the object schema");
    result = GetWhitespacePattern();
  }
  indent_manager_.EndIndent();
  --level_;
  return result;
}
std::string V41ToolCallingConverter::GenerateEnum(const EnumSpec& spec, const std::string& name) {
  std::vector<std::string> alternatives;
  for (const auto& value : spec.json_values) alternatives.push_back(GenerateConst({value}, name));
  return EBNFScriptCreator::Or(alternatives);
}
std::string V41ToolCallingConverter::GenerateAllOf(const AllOfSpec& spec, const std::string& name) {
  std::vector<SchemaSpecPtr> schemas;
  for (const auto& schema : spec.schemas) {
    if (std::holds_alternative<AnySpec>(schema->spec)) continue;
    if (std::none_of(schemas.begin(), schemas.end(), [&](const auto& previous) {
          return previous == schema || (!previous->cache_key.empty() && previous->cache_key == schema->cache_key);
        })) schemas.push_back(schema);
  }
  if (schemas.empty()) return GenerateAny(AnySpec{}, name);
  if (schemas.size() == 1) return GenerateFromSpec(schemas.front(), name);
  std::optional<FSMWithStartEnd> intersection;
  for (const auto& schema : schemas) {
    V41ToolCallingConverter child(resolver_);
    child.level_ = level_;
    child.key_context_ = key_context_;
    child.regular_context_ = true;
    auto fsm = V41RegularFSM(Grammar::FromEBNF(child.Convert(schema)));
    intersection = intersection ? V41Intersect(*intersection, fsm) : std::move(fsm);
  }
  if (V41Empty(*intersection)) throw EmptyIntersection();
  return V41EmitFSM(*intersection, ebnf_script_creator_);
}
std::string V41ToolCallingConverter::GeneratePatternObject(const ObjectSpec& spec,
    const std::string& name) {
  auto name_spec = spec.property_names ? spec.property_names : SchemaSpec::Make(StringSpec{});
  auto key_fsm = [&](const SchemaSpecPtr& schema) {
    V41ToolCallingConverter child(resolver_);
    child.level_ = 1;
    child.key_context_ = true;
    return V41RegularFSM(Grammar::FromEBNF(child.Convert(schema)));
  };
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);
  auto compile = [&](const FSMWithStartEnd& fsm) {
    EBNFScriptCreator script;
    auto root = script.AllocateRuleName("root");
    auto body = V41EmitFSM(fsm, script);
    script.AddRuleWithAllocatedName(root, body);
    return compiler.CompileGrammar(script.GetScript());
  };
  auto matches = [](const CompiledGrammar& grammar, const std::string& key) {
    GrammarMatcher matcher(grammar);
    return matcher.AcceptString(key) && matcher.IsCompleted();
  };
  auto allowed = key_fsm(name_spec);
  auto allowed_matcher = compile(allowed);
  std::vector<FSMWithStartEnd> patterns;
  std::vector<CompiledGrammar> pattern_matchers;
  for (const auto& property : spec.pattern_properties) {
    StringSpec pattern; pattern.pattern = property.pattern;
    patterns.push_back(key_fsm(SchemaSpec::Make(pattern)));
    pattern_matchers.push_back(compile(patterns.back()));
  }
  SchemaSpecPtr additional;
  if (spec.allow_additional_properties) additional = spec.additional_properties_schema;
  else if (spec.allow_unevaluated_properties) additional = spec.unevaluated_properties_schema;
  if (!additional && (spec.allow_additional_properties || spec.allow_unevaluated_properties))
    additional = SchemaSpec::Make(AnySpec{});

  auto fixed = spec.properties;
  std::vector<std::string> missing;
  for (const auto& required : spec.required)
    if (std::none_of(fixed.begin(), fixed.end(), [&](const auto& p) { return p.name == required; }))
      missing.push_back(required);
  std::sort(missing.begin(), missing.end());
  for (const auto& required : missing) fixed.push_back({required, nullptr});
  std::vector<ObjectSpec::Property> properties;
  for (const auto& property : fixed) {
    auto key = Replace(SafeJSON(picojson::value(property.name)), " ", "\\u0020");
    if (!matches(allowed_matcher, key)) {
      if (spec.required.count(property.name))
        throw std::invalid_argument("required parameter violates propertyNames: " + property.name);
      continue;
    }
    AllOfSpec value;
    if (property.schema) value.schemas.push_back(property.schema);
    for (size_t i = 0; i < patterns.size(); ++i)
      if (matches(pattern_matchers[i], key)) value.schemas.push_back(spec.pattern_properties[i].schema);
    if (value.schemas.empty()) {
      if (!additional) throw std::invalid_argument("required parameter matches no allowed property: " + property.name);
      value.schemas.push_back(additional);
    }
    properties.push_back({property.name, SchemaSpec::Make(value)});
  }
  // Required names materialized above are fixed too, so a dynamic branch must
  // never supply another occurrence with weaker constraints.
  if (!fixed.empty()) {
    EnumSpec excluded;
    for (const auto& property : fixed)
      excluded.json_values.push_back(picojson::value(property.name).serialize(false));
    allowed = V41Subtract(allowed, key_fsm(SchemaSpec::Make(excluded)));
  }
  struct Region { FSMWithStartEnd keys; AllOfSpec value; };
  std::vector<Region> regions;
  if (!V41Empty(allowed)) regions.push_back({std::move(allowed), {}});
  for (size_t i = 0; i < patterns.size(); ++i) {
    std::vector<Region> next;
    for (const auto& region : regions) {
      auto yes = V41Intersect(region.keys, patterns[i]);
      auto no = V41Subtract(region.keys, patterns[i]);
      if (!V41Empty(yes)) {
        auto value = region.value;
        value.schemas.push_back(spec.pattern_properties[i].schema);
        next.push_back({std::move(yes), std::move(value)});
      }
      if (!V41Empty(no)) next.push_back({std::move(no), region.value});
      if (next.size() > 256)
        throw std::invalid_argument("patternProperties exceeds the disjoint-region budget");
    }
    regions = std::move(next);
  }
  ++level_;
  indent_manager_.StartIndent();
  for (auto it = properties.begin(); it != properties.end();) {
    it->schema->cache_key = ebnf_script_creator_.AllocateRuleName("v41_pattern_cache");
    try {
      auto rule = CreateRule(it->schema, name + "_fixed_pattern_value");
      AddCache(it->schema->cache_key, rule);
      ++it;
    } catch (const EmptyIntersection&) {
      if (spec.required.count(it->name))
        throw std::invalid_argument("required parameter has incompatible value constraints: " + it->name);
      it = properties.erase(it);
    }
  }
  std::vector<std::string> dynamic;
  for (auto& region : regions) {
    if (region.value.schemas.empty()) {
      if (!additional) continue;
      region.value.schemas.push_back(additional);
    }
    std::string value;
    try {
      value = CreateRule(SchemaSpec::Make(region.value), name + "_pattern_value");
    } catch (const EmptyIntersection&) { continue; }
    auto key = V41EmitFSM(region.keys, ebnf_script_creator_);
    dynamic.push_back(FormatOtherProperty(key, value, name, "pattern"));
  }
  auto other = dynamic.empty() ? std::string{} : EBNFScriptCreator::Or(dynamic);
  auto dynamic_schema = dynamic.empty() ? nullptr : SchemaSpec::Make(AnySpec{});
  std::string result;
  if (!dynamic_schema && spec.min_properties > static_cast<int>(properties.size()))
    throw std::invalid_argument("patternProperties leaves too few allowed parameters");
  if (!properties.empty()) {
    result = GetPartialRuleForProperties(properties, spec.required, dynamic_schema, name,
        "pattern", spec.min_properties, spec.max_properties, other);
    if (spec.required.empty() && spec.min_properties == 0)
      result = "(" + result + ") | " + GetWhitespacePattern();
  } else if (dynamic_schema && spec.max_properties != 0) {
    result = GetWhitespacePattern() + " " + EBNFScriptCreator::Repeat(
        "(" + other + " " + GetWhitespacePattern() + ")", spec.min_properties, spec.max_properties);
  } else result = GetWhitespacePattern();
  indent_manager_.EndIndent();
  --level_;
  return result;
}
std::string V41ToolCallingConverter::GenerateRef(const RefSpec& spec, const std::string&) {
  auto key = ContextKey(spec.uri);
  if (auto found = refs_.find(key); found != refs_.end()) return found->second;
  auto name = ebnf_script_creator_.AllocateRuleName("v41_ref");
  refs_[key] = name;
  auto resolved = resolver_(spec.uri, name);
  auto body = GenerateFromSpec(resolved, name);
  ebnf_script_creator_.AddRuleWithAllocatedName(name, body);
  return name;
}
std::string V41ToolCallingConverter::FormatProperty(const std::string& key, const std::string& value,
    const std::string& name, int64_t index) {
  if (level_ != 1) return JSONSchemaConverter::FormatProperty(key, value, name, index);
  return Literal("<｜DSML｜ parameter name=" + Key(key) + " string=\"") + " " +
      value + " " + Literal(kEnd);
}
std::string V41ToolCallingConverter::FormatOtherProperty(const std::string& key, const std::string& value,
    const std::string& name, const std::string& suffix) {
  if (level_ != 1) return JSONSchemaConverter::FormatOtherProperty(key, value, name, suffix);
  return Literal("<｜DSML｜ parameter name=") + " " + key + " " + Literal(" string=\"") +
      " " + value + " " + Literal(kEnd);
}
std::string V41ToolCallingConverter::GetKeyPattern() const {
  return level_ == 1 ? kKey : JSONSchemaConverter::GetKeyPattern();
}
std::string V41ToolCallingConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& name) {
  return level_ == 1 ? NamePatternExcluding(SchemaSpec::Make(StringSpec{}), properties)
      : JSONSchemaConverter::GetKeyPatternExcluding(properties, name);
}
std::string V41ToolCallingConverter::NamePatternExcluding(const SchemaSpecPtr& spec,
    const std::vector<ObjectSpec::Property>& properties) {
  V41ToolCallingConverter names(resolver_);
  names.key_context_ = true;
  names.level_ = 1;
  auto allowed = V41RegularFSM(Grammar::FromEBNF(names.Convert(spec)));
  if (!properties.empty()) {
    EnumSpec excluded;
    for (const auto& property : properties)
      excluded.json_values.push_back(picojson::value(property.name).serialize(false));
    V41ToolCallingConverter fixed(resolver_);
    fixed.key_context_ = true;
    fixed.level_ = 1;
    allowed = V41Subtract(allowed,
        V41RegularFSM(Grammar::FromEBNF(fixed.Convert(SchemaSpec::Make(excluded)))));
  }
  return V41EmitFSM(allowed, ebnf_script_creator_);
}
std::string V41ToolCallingConverter::NextSeparator(bool end) {
  return level_ == 1 ? GetWhitespacePattern() : JSONSchemaConverter::NextSeparator(end);
}
std::string V41ToolSchemaToEBNF(const picojson::value& schema, bool strict) {
  return JSONSchemaToEBNF(schema, true, std::nullopt, std::nullopt, strict, 16,
      static_cast<JSONFormat>(5), false);
}
}  // namespace xgrammar

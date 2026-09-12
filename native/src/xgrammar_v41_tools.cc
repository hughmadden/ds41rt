#include "xgrammar_v41_tools.h"
#include "regex_converter.h"
#include <stdexcept>

namespace xgrammar {
namespace {
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
}  // namespace
V41ToolCallingConverter::V41ToolCallingConverter(RefResolver resolver)
    : JSONSchemaConverter(std::nullopt, std::nullopt, true, 16, resolver, false),
      resolver_(std::move(resolver)) {}
std::string V41ToolCallingConverter::ContextKey(const std::string& key) const {
  return std::to_string(std::min(level_, 2)) + ":" + key;
}
void V41ToolCallingConverter::AddCache(const std::string& key, const std::string& value) {
  if (!key.empty()) cache_[ContextKey(key)] = value;
}
std::optional<std::string> V41ToolCallingConverter::GetCache(const std::string& key) const {
  auto found = cache_.find(ContextKey(key));
  return found == cache_.end() ? std::nullopt : std::optional<std::string>(found->second);
}
void V41ToolCallingConverter::AddBasicRules() {
  level_ = 2;
  JSONSchemaConverter::AddBasicRules();
  level_ = 0;
  ebnf_script_creator_.AddRule(kRaw,
      "TagDispatch(loop_after_dispatch=false,excludes=(" + Literal(kEndPrefix) + "))");
  // JSON-quoted attribute names, without raw spaces or control characters.
  ebnf_script_creator_.AddRule(kKey, R"ebnf("\"" ([^\x00-\x20"\\] | "\\" basic_escape)* "\"")ebnf");
}
std::string V41ToolCallingConverter::Flag(const std::string& value, bool string) const {
  if (level_ != 1) return value;
  auto body = "(" + value + ")";
  return Literal(string ? "true\">" : "false\">") + " " +
      (string ? body : GetWhitespacePattern() + " " + body + " " + GetWhitespacePattern());
}
#define V41_SCALAR(Name, Spec) \
std::string V41ToolCallingConverter::Generate##Name(const Spec& spec, const std::string& name) { \
  return Flag(JSONSchemaConverter::Generate##Name(spec, name)); \
}
V41_SCALAR(Integer, IntegerSpec)
V41_SCALAR(Number, NumberSpec)
V41_SCALAR(Boolean, BooleanSpec)
V41_SCALAR(Null, NullSpec)
#undef V41_SCALAR
std::string V41ToolCallingConverter::GenerateString(const StringSpec& spec, const std::string& name) {
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
  return Flag(body, true);
}
std::string V41ToolCallingConverter::GenerateArray(const ArraySpec& spec, const std::string& name) {
  ++level_;
  auto value = JSONSchemaConverter::GenerateArray(spec, name);
  --level_;
  return Flag(value);
}
std::string V41ToolCallingConverter::GenerateObject(const ObjectSpec& spec, const std::string& name, bool) {
  const bool root = level_ == 0;
  if (root && (!spec.pattern_properties.empty() || spec.property_names)) {
    throw std::invalid_argument("V4.1 root patternProperties/propertyNames conversion is not implemented");
  }
  ++level_;
  auto value = JSONSchemaConverter::GenerateObject(spec, name, !root);
  --level_;
  return Flag(value);
}
std::string V41ToolCallingConverter::GenerateAny(const AnySpec& spec, const std::string& name) {
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
std::string V41ToolCallingConverter::GenerateEnum(const EnumSpec& spec, const std::string& name) {
  std::vector<std::string> alternatives;
  for (const auto& value : spec.json_values) alternatives.push_back(GenerateConst({value}, name));
  return EBNFScriptCreator::Or(alternatives);
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
  return level_ == 1 ? kKey : JSONSchemaConverter::GetKeyPatternExcluding(properties, name);
}
std::string V41ToolCallingConverter::NextSeparator(bool end) {
  return level_ == 1 ? GetWhitespacePattern() : JSONSchemaConverter::NextSeparator(end);
}
std::string V41ToolSchemaToEBNF(const picojson::value& schema, bool strict) {
  return JSONSchemaToEBNF(schema, true, std::nullopt, std::nullopt, strict, 16,
      static_cast<JSONFormat>(5), false);
}
}  // namespace xgrammar

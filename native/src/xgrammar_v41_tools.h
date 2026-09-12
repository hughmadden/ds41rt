#pragma once
#include "json_schema_converter.h"

namespace xgrammar {
// Project extension of the pinned schema IR. The build adds a factory case in
// a generated copy of the upstream converter; the vendored source stays intact.
class V41ToolCallingConverter : public JSONSchemaConverter {
 public:
  explicit V41ToolCallingConverter(RefResolver resolver);
 protected:
  std::string GenerateInteger(const IntegerSpec&, const std::string&) override;
  std::string GenerateNumber(const NumberSpec&, const std::string&) override;
  std::string GenerateString(const StringSpec&, const std::string&) override;
  std::string GenerateBoolean(const BooleanSpec&, const std::string&) override;
  std::string GenerateNull(const NullSpec&, const std::string&) override;
  std::string GenerateArray(const ArraySpec&, const std::string&) override;
  std::string GenerateObject(const ObjectSpec&, const std::string&, bool = true) override;
  std::string GenerateAny(const AnySpec&, const std::string&) override;
  std::string GenerateConst(const ConstSpec&, const std::string&) override;
  std::string GenerateEnum(const EnumSpec&, const std::string&) override;
  std::string GenerateRef(const RefSpec&, const std::string&) override;
  std::string FormatProperty(const std::string&, const std::string&, const std::string&, int64_t) override;
  std::string FormatOtherProperty(const std::string&, const std::string&, const std::string&, const std::string&) override;
  std::string GetKeyPattern() const override;
  std::string GetKeyPatternExcluding(const std::vector<ObjectSpec::Property>&, const std::string&) override;
  std::string NextSeparator(bool = false) override;
  void AddBasicRules() override;
  void AddCache(const std::string&, const std::string&) override;
  std::optional<std::string> GetCache(const std::string&) const override;
 private:
  int level_ = 0;
  bool key_context_ = false;
  RefResolver resolver_;
  std::unordered_map<std::string, std::string> cache_, refs_;
  std::string ContextKey(const std::string&) const;
  std::string Flag(const std::string&, bool string = false) const;
  std::string EncodedString(const StringSpec&);
  std::string GenerateNamedObject(const ObjectSpec&, const std::string&);
};
std::string V41ToolSchemaToEBNF(const picojson::value& schema, bool strict);
}  // namespace xgrammar

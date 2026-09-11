# Both official gate geometries share runtime-row AOT entry points.
set(DS41RT_V41_ROUTER_DIR "${CMAKE_CURRENT_BINARY_DIR}/v41_router")
set(DS41RT_V41_ROUTER_OUTPUTS "${DS41RT_V41_ROUTER_DIR}/v41_router.json" "${DS41RT_V41_ROUTER_DIR}/v41_router_dispatch.h")
set(DS41RT_V41_ROUTER_OBJECTS)
foreach(experts IN ITEMS 128 384)
  list(APPEND DS41RT_V41_ROUTER_OBJECTS "${DS41RT_V41_ROUTER_DIR}/v41_router_e${experts}.o")
  list(APPEND DS41RT_V41_ROUTER_OUTPUTS "${DS41RT_V41_ROUTER_DIR}/v41_router_e${experts}.h")
endforeach()
list(APPEND DS41RT_V41_ROUTER_OUTPUTS ${DS41RT_V41_ROUTER_OBJECTS})
add_custom_command(
  OUTPUT ${DS41RT_V41_ROUTER_OUTPUTS}
  COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${DS41RT_SPARKINFER_PYTHON_ENV}
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_router_aot.py"
    --output-dir "${DS41RT_V41_ROUTER_DIR}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_router_aot.py"
    ${DS41RT_SPARKINFER_PROVENANCE_INPUTS} ${DS41RT_SPARKINFER_EXPORT_INPUTS}
  VERBATIM)
add_custom_target(ds41rt_v41_router_export DEPENDS ${DS41RT_V41_ROUTER_OUTPUTS})
add_dependencies(ds41rt_v41_router_export ds41rt_verify_sparkinfer_source)
set_source_files_properties(${DS41RT_V41_ROUTER_OBJECTS} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND DS41RT_NATIVE_SOURCES ${DS41RT_V41_ROUTER_OBJECTS} src/v41_router_scores.cc)

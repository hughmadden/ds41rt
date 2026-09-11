# The export executes on the target architecture and consumes b12x's own planner.
if(NOT DS41RT_ENABLE_CUDA)
  message(FATAL_ERROR "V4.1 expert AOT requires CUDA")
endif()
if(DS41RT_CUDA_ARCHITECTURES STREQUAL "120" OR DS41RT_CUDA_ARCHITECTURES STREQUAL "120f")
  set(DS41RT_V41_EXPERT_ROLE coordinator)
  set(DS41RT_V41_EXPERT_INPUT_FORMAT bf16)
elseif(DS41RT_CUDA_ARCHITECTURES STREQUAL "121")
  set(DS41RT_V41_EXPERT_ROLE spark)
  set(DS41RT_V41_EXPERT_INPUT_FORMAT fp8_k32)
else()
  message(FATAL_ERROR "V4.1 expert AOT requires one native SM120 or SM121 target")
endif()
set(DS41RT_V41_EXPERT_SLICE_WIDTH "" CACHE STRING
  "Experimental Spark fused-slice width (64, 128, 192); empty keeps qualified serving backend")
set(DS41RT_V41_EXPERT_EXPORT_SCRIPT export_b12x_v41_experts_aot.py)
set(DS41RT_V41_EXPERT_EXPORT_ARGS
  --role "${DS41RT_V41_EXPERT_ROLE}" --input-format "${DS41RT_V41_EXPERT_INPUT_FORMAT}")
if(NOT DS41RT_V41_EXPERT_SLICE_WIDTH STREQUAL "")
  if(NOT DS41RT_V41_EXPERT_ROLE STREQUAL "spark" OR
      NOT DS41RT_V41_EXPERT_SLICE_WIDTH MATCHES "^(64|128|192)$")
    message(FATAL_ERROR "Experimental expert slices require Spark and width 64, 128 or 192")
  endif()
  set(DS41RT_V41_EXPERT_EXPORT_SCRIPT export_b12x_v41_slices_aot.py)
  set(DS41RT_V41_EXPERT_EXPORT_ARGS --width "${DS41RT_V41_EXPERT_SLICE_WIDTH}"
    --rows "1,16,80,256,1024,4096")
endif()
set(DS41RT_V41_EXPERT_DIR "${CMAKE_CURRENT_BINARY_DIR}/v41_experts")
set(DS41RT_V41_EXPERT_OBJECTS)
set(DS41RT_V41_EXPERT_HEADERS
  "${DS41RT_V41_EXPERT_DIR}/v41_expert_input_quant.h"
  "${DS41RT_V41_EXPERT_DIR}/v41_input_quant_dispatch.h")
list(APPEND DS41RT_V41_EXPERT_OBJECTS "${DS41RT_V41_EXPERT_DIR}/v41_expert_input_quant.o")
foreach(rows IN ITEMS 1 16 80 256 1024 4096)
  if(DS41RT_V41_EXPERT_SLICE_WIDTH STREQUAL "")
    set(stem "${DS41RT_V41_EXPERT_DIR}/v41_${DS41RT_V41_EXPERT_ROLE}_m${rows}")
  else()
    set(stem "${DS41RT_V41_EXPERT_DIR}/v41_slices_m${rows}_w${DS41RT_V41_EXPERT_SLICE_WIDTH}")
  endif()
  list(APPEND DS41RT_V41_EXPERT_OBJECTS "${stem}.o")
  list(APPEND DS41RT_V41_EXPERT_HEADERS "${stem}.h")
endforeach()
add_custom_command(
  OUTPUT "${DS41RT_V41_EXPERT_DIR}/v41_experts.json"
    "${DS41RT_V41_EXPERT_DIR}/v41_expert_variants.h"
    ${DS41RT_V41_EXPERT_OBJECTS} ${DS41RT_V41_EXPERT_HEADERS}
  COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${DS41RT_SPARKINFER_PYTHON_ENV}
    "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/${DS41RT_V41_EXPERT_EXPORT_SCRIPT}"
    --output-dir "${DS41RT_V41_EXPERT_DIR}" ${DS41RT_V41_EXPERT_EXPORT_ARGS}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/${DS41RT_V41_EXPERT_EXPORT_SCRIPT}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_experts_aot.py"
    ${DS41RT_SPARKINFER_PROVENANCE_INPUTS} ${DS41RT_SPARKINFER_EXPORT_INPUTS}
  COMMENT "Exporting native V4.1 expert kernels and scratch layouts"
  VERBATIM
)
add_custom_target(ds41rt_v41_experts_export DEPENDS
  "${DS41RT_V41_EXPERT_DIR}/v41_expert_variants.h"
  "${DS41RT_V41_EXPERT_DIR}/v41_experts.json"
  ${DS41RT_V41_EXPERT_OBJECTS} ${DS41RT_V41_EXPERT_HEADERS})
add_dependencies(ds41rt_v41_experts_export ds41rt_verify_sparkinfer_source)
set_source_files_properties(${DS41RT_V41_EXPERT_OBJECTS} PROPERTIES
  EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND DS41RT_NATIVE_SOURCES ${DS41RT_V41_EXPERT_OBJECTS} src/v41_experts.cc)

cmake_minimum_required(VERSION 3.24)
project(ds4rt_native LANGUAGES C CXX)

option(DS4RT_ENABLE_CUDA "Build CUDA kernels and CUDA-backed ABI functions" OFF)
option(DS4RT_ENABLE_RDMA "Build RDMA helper components when libibverbs is available" OFF)
option(DS4RT_ENABLE_SPARKINFER_AOT "Build Spark-side SparkInfer AOT kernels" OFF)
option(DS4RT_ENABLE_DS4_FLASH_AOT
  "Build DeepSeek-V4-Flash Spark expert-TP4 SparkInfer AOT kernels" OFF)
option(DS4RT_ENABLE_SPARKINFER_COORDINATOR_AOT
  "Build coordinator SparkInfer AOT kernels" OFF)
option(DS4RT_ENABLE_B12X_AOT
  "Deprecated alias for DS4RT_ENABLE_SPARKINFER_AOT" OFF)
option(DS4RT_ENABLE_B12X_COORDINATOR_AOT
  "Deprecated alias for DS4RT_ENABLE_SPARKINFER_COORDINATOR_AOT" OFF)
option(DS4RT_ENABLE_W8A16_AOT "Build coordinator row-major W8A16 Triton AOT kernels" OFF)
option(DS4RT_ENABLE_FLASHINFER_PACKED_MLA
  "Build the exact grouped-split SM120 packed-FP8 MLA kernel" OFF)
option(DS4RT_ENABLE_NCCL "Build multi-node NCCL helper components" OFF)
option(DS4RT_ENABLE_XGRAMMAR "Build constrained-decoding grammar support" ON)
set(DS4RT_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA architectures for native kernels")
set(DS4RT_SPARKINFER_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sparkinfer"
  CACHE PATH "Pinned SparkInfer source tree used by every SparkInfer-backed AOT export")
set(DS4RT_SPARKINFER_LOCK_FILE
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sparkinfer.lock.json"
  CACHE FILEPATH "Provenance lock for DS4RT_SPARKINFER_SOURCE_DIR")
set(DS4RT_XGRAMMAR_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xgrammar"
  CACHE PATH "Pinned XGrammar source tree")
set(DS4RT_XGRAMMAR_LOCK_FILE
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xgrammar.lock.json"
  CACHE FILEPATH "Provenance lock for DS4RT_XGRAMMAR_SOURCE_DIR")

if(DS4RT_ENABLE_B12X_AOT)
  message(DEPRECATION
    "DS4RT_ENABLE_B12X_AOT is deprecated; use DS4RT_ENABLE_SPARKINFER_AOT")
  set(DS4RT_ENABLE_SPARKINFER_AOT ON)
endif()
if(DS4RT_ENABLE_B12X_COORDINATOR_AOT)
  message(DEPRECATION
    "DS4RT_ENABLE_B12X_COORDINATOR_AOT is deprecated; "
    "use DS4RT_ENABLE_SPARKINFER_COORDINATOR_AOT")
  set(DS4RT_ENABLE_SPARKINFER_COORDINATOR_AOT ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(DS4RT_NATIVE_SOURCES src/ds4rt_native.cc)

if(DS4RT_ENABLE_XGRAMMAR)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  if(NOT EXISTS "${DS4RT_XGRAMMAR_SOURCE_DIR}/include/xgrammar/compiler.h" OR
     NOT EXISTS "${DS4RT_XGRAMMAR_SOURCE_DIR}/3rdparty/dlpack/include/dlpack/dlpack.h" OR
     NOT EXISTS "${DS4RT_XGRAMMAR_SOURCE_DIR}/3rdparty/picojson/picojson.h")
    message(FATAL_ERROR
      "Pinned XGrammar source is incomplete under ${DS4RT_XGRAMMAR_SOURCE_DIR}; "
      "initialize third_party/xgrammar and its nested submodules")
  endif()
  set(DS4RT_XGRAMMAR_VERIFY_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/verify-xgrammar-source.py")
  if(NOT EXISTS "${DS4RT_XGRAMMAR_VERIFY_SCRIPT}")
    message(FATAL_ERROR
      "XGrammar source verifier was not found: ${DS4RT_XGRAMMAR_VERIFY_SCRIPT}")
  endif()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${DS4RT_XGRAMMAR_VERIFY_SCRIPT}"
      --source "${DS4RT_XGRAMMAR_SOURCE_DIR}"
      --lock "${DS4RT_XGRAMMAR_LOCK_FILE}"
    RESULT_VARIABLE DS4RT_XGRAMMAR_VERIFY_RESULT
    OUTPUT_VARIABLE DS4RT_XGRAMMAR_VERIFY_OUTPUT
    ERROR_VARIABLE DS4RT_XGRAMMAR_VERIFY_ERROR
  )
  if(NOT DS4RT_XGRAMMAR_VERIFY_RESULT EQUAL 0)
    message(FATAL_ERROR
      "XGrammar source verification failed:\n${DS4RT_XGRAMMAR_VERIFY_ERROR}")
  endif()
  string(STRIP "${DS4RT_XGRAMMAR_VERIFY_OUTPUT}" DS4RT_XGRAMMAR_VERIFY_OUTPUT)
  message(STATUS "${DS4RT_XGRAMMAR_VERIFY_OUTPUT}")
  file(GLOB_RECURSE DS4RT_XGRAMMAR_SOURCES
    CONFIGURE_DEPENDS
    "${DS4RT_XGRAMMAR_SOURCE_DIR}/cpp/*.cc"
  )
  list(FILTER DS4RT_XGRAMMAR_SOURCES EXCLUDE REGEX "/cpp/tvm_ffi/.*\\.cc$")
  add_library(ds4rt_xgrammar STATIC ${DS4RT_XGRAMMAR_SOURCES})
  target_include_directories(ds4rt_xgrammar
    PUBLIC "${DS4RT_XGRAMMAR_SOURCE_DIR}/include"
  )
  target_include_directories(ds4rt_xgrammar SYSTEM PUBLIC
    "${DS4RT_XGRAMMAR_SOURCE_DIR}/3rdparty/dlpack/include"
    "${DS4RT_XGRAMMAR_SOURCE_DIR}/3rdparty/picojson"
  )
  target_compile_definitions(ds4rt_xgrammar PUBLIC
    XGRAMMAR_ENABLE_CPPTRACE=0
    XGRAMMAR_ENABLE_INTERNAL_CHECK=0
  )
  find_package(Threads REQUIRED)
  target_link_libraries(ds4rt_xgrammar PUBLIC Threads::Threads)
  list(APPEND DS4RT_NATIVE_SOURCES src/xgrammar_adapter.cc)
endif()

set(DS4RT_BUILD_SPARKINFER_AOT OFF)
if(DS4RT_ENABLE_SPARKINFER_AOT OR DS4RT_ENABLE_DS4_FLASH_AOT OR
   DS4RT_ENABLE_SPARKINFER_COORDINATOR_AOT OR DS4RT_ENABLE_W8A16_AOT)
  set(DS4RT_BUILD_SPARKINFER_AOT ON)
endif()
if(DS4RT_BUILD_SPARKINFER_AOT)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  if(NOT EXISTS "${DS4RT_SPARKINFER_SOURCE_DIR}/b12x/__init__.py")
    message(FATAL_ERROR
      "SparkInfer was not found under DS4RT_SPARKINFER_SOURCE_DIR="
      "${DS4RT_SPARKINFER_SOURCE_DIR}. Initialize third_party/sparkinfer or "
      "select an explicit pinned source tree.")
  endif()
  set(DS4RT_SPARKINFER_VERIFY_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/verify-sparkinfer-source.py")
  set(DS4RT_SPARKINFER_PYTHON_BOOTSTRAP
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/_pinned_sparkinfer.py")
  if(NOT EXISTS "${DS4RT_SPARKINFER_VERIFY_SCRIPT}")
    message(FATAL_ERROR
      "SparkInfer source verifier was not found: ${DS4RT_SPARKINFER_VERIFY_SCRIPT}")
  endif()
  if(NOT EXISTS "${DS4RT_SPARKINFER_PYTHON_BOOTSTRAP}")
    message(FATAL_ERROR
      "SparkInfer Python bootstrap was not found: "
      "${DS4RT_SPARKINFER_PYTHON_BOOTSTRAP}")
  endif()
  set(DS4RT_SPARKINFER_VERIFY_COMMAND
    "${Python3_EXECUTABLE}" "${DS4RT_SPARKINFER_VERIFY_SCRIPT}"
    --source "${DS4RT_SPARKINFER_SOURCE_DIR}"
    --lock "${DS4RT_SPARKINFER_LOCK_FILE}"
  )
  set(DS4RT_SPARKINFER_PYTHON_ENV
    "DS4RT_SPARKINFER_SOURCE_DIR=${DS4RT_SPARKINFER_SOURCE_DIR}"
    "DS4RT_SPARKINFER_LOCK_FILE=${DS4RT_SPARKINFER_LOCK_FILE}"
    # Persisted objects are executable-only ExternalBinaryModules. AOT export
    # needs the live compiler IR consumed by export_to_c(), so never substitute
    # a runtime disk-cache hit in these build-time Python processes.
    "B12X_COMPILE_DISK_CACHE=0"
    "PYTHONDONTWRITEBYTECODE=1"
    "PYTHONPATH=${DS4RT_SPARKINFER_SOURCE_DIR}"
  )
  execute_process(
    COMMAND ${DS4RT_SPARKINFER_VERIFY_COMMAND}
    RESULT_VARIABLE DS4RT_SPARKINFER_VERIFY_RESULT
    OUTPUT_VARIABLE DS4RT_SPARKINFER_VERIFY_OUTPUT
    ERROR_VARIABLE DS4RT_SPARKINFER_VERIFY_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT DS4RT_SPARKINFER_VERIFY_RESULT EQUAL 0)
    message(FATAL_ERROR
      "Pinned SparkInfer verification failed:\n${DS4RT_SPARKINFER_VERIFY_ERROR}")
  endif()
  message(STATUS "${DS4RT_SPARKINFER_VERIFY_OUTPUT}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${DS4RT_SPARKINFER_LOCK_FILE}"
    "${DS4RT_SPARKINFER_VERIFY_SCRIPT}"
  )
  set(DS4RT_SPARKINFER_PROVENANCE_INPUTS
    "${DS4RT_SPARKINFER_LOCK_FILE}"
    "${DS4RT_SPARKINFER_VERIFY_SCRIPT}"
    "${DS4RT_SPARKINFER_PYTHON_BOOTSTRAP}"
  )
  file(GLOB_RECURSE DS4RT_SPARKINFER_EXPORT_INPUTS
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${DS4RT_SPARKINFER_SOURCE_DIR}/b12x/*.py"
    "${DS4RT_SPARKINFER_SOURCE_DIR}/b12x/*.cu"
    "${DS4RT_SPARKINFER_SOURCE_DIR}/b12x/*.cuh"
    "${DS4RT_SPARKINFER_SOURCE_DIR}/b12x/*.h"
    "${DS4RT_SPARKINFER_SOURCE_DIR}/b12x/*.hpp"
  )
  list(APPEND DS4RT_SPARKINFER_EXPORT_INPUTS
    "${DS4RT_SPARKINFER_SOURCE_DIR}/pyproject.toml"
  )
  add_custom_target(ds4rt_verify_sparkinfer_source
    COMMAND ${DS4RT_SPARKINFER_VERIFY_COMMAND}
    COMMENT "Verifying pinned SparkInfer source"
    VERBATIM
  )
endif()

if(DS4RT_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND DS4RT_NATIVE_SOURCES
    cuda/kernels/norm.cu
    cuda/kernels/mlp.cu
    cuda/kernels/residual.cu
    cuda/kernels/kv_cache.cu
    cuda/kernels/mla_indexing.cu
    cuda/kernels/router.cu
    cuda/kernels/linear.cu
    cuda/kernels/attention.cu
    cuda/kernels/embedding.cu
    cuda/kernels/sampling.cu
    cuda/kernels/nibbles.cu
    cuda/kernels/nvfp4_payload.cu
    cuda/kernels/ds4_flash_direct.cu
  )
endif()

set(DS4RT_BUILD_FLASHINFER_PACKED_MLA OFF)
if(DS4RT_ENABLE_FLASHINFER_PACKED_MLA OR DS4RT_ENABLE_SPARKINFER_COORDINATOR_AOT OR DS4RT_ENABLE_W8A16_AOT)
  set(DS4RT_BUILD_FLASHINFER_PACKED_MLA ON)
endif()

if(DS4RT_BUILD_FLASHINFER_PACKED_MLA)
  if(NOT DS4RT_ENABLE_CUDA)
    message(FATAL_ERROR "DS4RT_ENABLE_FLASHINFER_PACKED_MLA=ON requires DS4RT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "import pathlib, flashinfer; print(pathlib.Path(flashinfer.__file__).resolve().parent / 'data' / 'include')"
    OUTPUT_VARIABLE DS4RT_FLASHINFER_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  if(NOT EXISTS "${DS4RT_FLASHINFER_INCLUDE_DIR}/flashinfer/attention/sparse_mla_sm120/decode_dsv3_2_kernel.cuh")
    message(FATAL_ERROR
      "FlashInfer SM120 sparse-MLA headers were not found under ${DS4RT_FLASHINFER_INCLUDE_DIR}")
  endif()
  list(APPEND DS4RT_NATIVE_SOURCES
    cuda/kernels/packed_fp8_mla_exact.cu
  )
  set_source_files_properties(cuda/kernels/packed_fp8_mla_exact.cu PROPERTIES
    COMPILE_OPTIONS "-use_fast_math;-O3"
  )
  if(DS4RT_CUDA_ARCHITECTURES STREQUAL "120")
    set(DS4RT_CUDA_ARCHITECTURES "120f")
  endif()
endif()

if(DS4RT_ENABLE_SPARKINFER_AOT)
  if(NOT DS4RT_ENABLE_CUDA)
    message(FATAL_ERROR "DS4RT_ENABLE_SPARKINFER_AOT=ON requires DS4RT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(DS4RT_B12X_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/b12x_aot")
  set(DS4RT_B12X_AOT_REGIMES 1 2 4 8 16 32 64 128 256)
  set(DS4RT_B12X_AOT_TOPK8_PREFILL_REGIMES 2 4 8 16 32 64 128 256 512 1024 2048)
  set(DS4RT_B12X_AOT_OBJECTS)
  set(DS4RT_B12X_AOT_HEADERS)
  list(APPEND DS4RT_B12X_AOT_OBJECTS
    "${DS4RT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1.o"
    "${DS4RT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1_fused_sum.o"
  )
  foreach(regime IN LISTS DS4RT_B12X_AOT_TOPK8_PREFILL_REGIMES)
    list(APPEND DS4RT_B12X_AOT_OBJECTS
      "${DS4RT_B12X_AOT_DIR}/moe_tp4_w4a16_prefill_m${regime}_topk8.o"
    )
  endforeach()
  foreach(regime IN LISTS DS4RT_B12X_AOT_REGIMES)
    list(APPEND DS4RT_B12X_AOT_OBJECTS
      "${DS4RT_B12X_AOT_DIR}/moe_tp4_w4a16_top1_m${regime}.o"
    )
  endforeach()
  list(APPEND DS4RT_B12X_AOT_HEADERS
    "${DS4RT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1.h"
    "${DS4RT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1_fused_sum.h"
    "${DS4RT_B12X_AOT_DIR}/b12x_spark_moe_aot_config.h"
  )
  foreach(regime IN LISTS DS4RT_B12X_AOT_TOPK8_PREFILL_REGIMES)
    list(APPEND DS4RT_B12X_AOT_HEADERS
      "${DS4RT_B12X_AOT_DIR}/moe_tp4_w4a16_prefill_m${regime}_topk8.h"
    )
  endforeach()
  foreach(regime IN LISTS DS4RT_B12X_AOT_REGIMES)
    list(APPEND DS4RT_B12X_AOT_HEADERS
      "${DS4RT_B12X_AOT_DIR}/moe_tp4_w4a16_top1_m${regime}.h"
    )
  endforeach()
  add_custom_command(
    OUTPUT
      "${DS4RT_B12X_AOT_DIR}/b12x_spark_aot.stamp"
      "${DS4RT_B12X_AOT_DIR}/b12x_spark_moe_aot.meta"
      ${DS4RT_B12X_AOT_OBJECTS}
      ${DS4RT_B12X_AOT_HEADERS}
    COMMAND ${DS4RT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${DS4RT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_moe_aot.py"
      --output-dir "${DS4RT_B12X_AOT_DIR}"
      --target-sms 48
    COMMAND "${CMAKE_COMMAND}" -E touch
      "${DS4RT_B12X_AOT_DIR}/b12x_spark_aot.stamp"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_moe_aot.py"
      ${DS4RT_SPARKINFER_PROVENANCE_INPUTS}
      ${DS4RT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting Spark-side SparkInfer AOT kernels"
    VERBATIM
  )
  add_custom_target(ds4rt_b12x_aot_export
    DEPENDS
      "${DS4RT_B12X_AOT_DIR}/b12x_spark_aot.stamp"
      "${DS4RT_B12X_AOT_DIR}/b12x_spark_moe_aot.meta"
      ${DS4RT_B12X_AOT_OBJECTS}
      ${DS4RT_B12X_AOT_HEADERS}
  )
  add_dependencies(ds4rt_b12x_aot_export ds4rt_verify_sparkinfer_source)
  set_source_files_properties(${DS4RT_B12X_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND DS4RT_NATIVE_SOURCES
    cuda/kernels/b12x_direct.cu
    ${DS4RT_B12X_AOT_OBJECTS}
  )

  set(DS4RT_B12X_M1_PARITY_AOT_REGIMES 2 3 4 5 6 7 8)
  set(DS4RT_B12X_M1_PARITY_AOT_OBJECTS)
  set(DS4RT_B12X_M1_PARITY_AOT_HEADERS
    "${DS4RT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot_config.h"
  )
  foreach(regime IN LISTS DS4RT_B12X_M1_PARITY_AOT_REGIMES)
    list(APPEND DS4RT_B12X_M1_PARITY_AOT_OBJECTS
      "${DS4RT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_m${regime}_topk8.o"
      "${DS4RT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_m${regime}_topk8.o"
      "${DS4RT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_wide_m${regime}_topk8.o"
    )
    list(APPEND DS4RT_B12X_M1_PARITY_AOT_HEADERS
      "${DS4RT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_m${regime}_topk8.h"
      "${DS4RT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_m${regime}_topk8.h"
      "${DS4RT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_wide_m${regime}_topk8.h"
    )
  endforeach()
  add_custom_command(
    OUTPUT
      "${DS4RT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.stamp"
      "${DS4RT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.meta"
      ${DS4RT_B12X_M1_PARITY_AOT_OBJECTS}
      ${DS4RT_B12X_M1_PARITY_AOT_HEADERS}
    COMMAND ${DS4RT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${DS4RT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_w4a16_m1_parity_aot.py"
      --output-dir "${DS4RT_B12X_AOT_DIR}"
      --target-sms 48
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_w4a16_m1_parity_aot.py"
      ${DS4RT_SPARKINFER_PROVENANCE_INPUTS}
      ${DS4RT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting ordered Spark W4A16 M=2..8 parity candidates"
    VERBATIM
  )
  add_custom_target(ds4rt_b12x_m1_parity_aot_export
    DEPENDS
      "${DS4RT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.stamp"
      "${DS4RT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.meta"
      ${DS4RT_B12X_M1_PARITY_AOT_OBJECTS}
      ${DS4RT_B12X_M1_PARITY_AOT_HEADERS}
  )
  add_dependencies(
    ds4rt_b12x_m1_parity_aot_export
    ds4rt_verify_sparkinfer_source
  )
  set_source_files_properties(${DS4RT_B12X_M1_PARITY_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND DS4RT_NATIVE_SOURCES ${DS4RT_B12X_M1_PARITY_AOT_OBJECTS})

endif()

if(DS4RT_ENABLE_DS4_FLASH_AOT)
  if(NOT DS4RT_ENABLE_CUDA)
    message(FATAL_ERROR "DS4RT_ENABLE_DS4_FLASH_AOT=ON requires DS4RT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(DS4RT_DS4_FLASH_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ds4_flash_aot")
  set(DS4RT_DS4_FLASH_AOT_PREFILL_REGIMES 2 4 8 16 32 64 128 256 512 1024 2048)
  set(DS4RT_DS4_FLASH_AOT_MIXED_EXACT_DECODE_REGIMES 3 5 6 7 9 10 11 12)
  set(DS4RT_DS4_FLASH_AOT_OBJECTS
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_w4a16_decode_m1_fused_sum.o"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m1.o"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m4_direct_m4.o"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m8_direct_m6.o"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m1.o"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m4_direct_m4.o"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m8_direct_m6.o"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_topk6_sum.o"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m1.o"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_up_m1.o"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_up_m8.o"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_down_m1.o"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_down_m8.o")
  set(DS4RT_DS4_FLASH_AOT_HEADERS
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_w4a16_decode_m1_fused_sum.h"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m1.h"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m4_direct_m4.h"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m8_direct_m6.h"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m1.h"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m4_direct_m4.h"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m8_direct_m6.h"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_topk6_sum.h"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m1.h"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_up_m1.h"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_up_m8.h"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_down_m1.h"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_down_m8.h"
    "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_spark_moe_aot_config.h"
  )
  foreach(regime IN LISTS DS4RT_DS4_FLASH_AOT_PREFILL_REGIMES)
    list(APPEND DS4RT_DS4_FLASH_AOT_OBJECTS
      "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_w4a16_prefill_m${regime}_topk6.o"
      "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_prefill_m${regime}_topk6.o"
      "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_prefill_m${regime}_topk6.o"
      "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m${regime}.o")
    list(APPEND DS4RT_DS4_FLASH_AOT_HEADERS
      "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_w4a16_prefill_m${regime}_topk6.h"
      "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_prefill_m${regime}_topk6.h"
      "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_prefill_m${regime}_topk6.h"
      "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m${regime}.h")
  endforeach()
  foreach(regime IN LISTS DS4RT_DS4_FLASH_AOT_MIXED_EXACT_DECODE_REGIMES)
    list(APPEND DS4RT_DS4_FLASH_AOT_OBJECTS
      "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m${regime}.o")
    list(APPEND DS4RT_DS4_FLASH_AOT_HEADERS
      "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m${regime}.h")
  endforeach()
  add_custom_command(
    OUTPUT
      "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_spark_aot.stamp"
      "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_spark_moe_aot.meta"
      ${DS4RT_DS4_FLASH_AOT_OBJECTS}
      ${DS4RT_DS4_FLASH_AOT_HEADERS}
    COMMAND ${DS4RT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${DS4RT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_ds4_flash_spark_moe_aot.py"
      --output-dir "${DS4RT_DS4_FLASH_AOT_DIR}"
      --target-sms 48
    COMMAND "${CMAKE_COMMAND}" -E touch
      "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_spark_aot.stamp"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_ds4_flash_spark_moe_aot.py"
      ${DS4RT_SPARKINFER_PROVENANCE_INPUTS}
      ${DS4RT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting DeepSeek-V4-Flash Spark expert-TP4 AOT kernel"
    VERBATIM
  )
  add_custom_target(ds4rt_ds4_flash_aot_export
    DEPENDS
      "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_spark_aot.stamp"
      "${DS4RT_DS4_FLASH_AOT_DIR}/ds4_flash_spark_moe_aot.meta"
      ${DS4RT_DS4_FLASH_AOT_OBJECTS}
      ${DS4RT_DS4_FLASH_AOT_HEADERS}
  )
  add_dependencies(ds4rt_ds4_flash_aot_export ds4rt_verify_sparkinfer_source)

  # Pro experts are trellis-only. Keep their generated symbols, resident
  # geometry, and native bridge separate from the Flash FP4-compatible path.
  set(DS4RT_DS4_PRO_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ds4_pro_aot")
  set(DS4RT_DS4_PRO_AOT_OBJECTS
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_decode_m1.o"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_topk6_sum.o"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m1.o"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m8.o"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m2048.o"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m1.o"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m8.o"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m2048.o"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_input_quant_m2048.o"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_activated_quant_m2048.o")
  set(DS4RT_DS4_PRO_AOT_HEADERS
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_decode_m1.h"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_topk6_sum.h"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m1.h"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m8.h"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m2048.h"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m1.h"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m8.h"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m2048.h"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_input_quant_m2048.h"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_activated_quant_m2048.h"
    "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_spark_moe_aot_config.h")
  foreach(regime IN LISTS DS4RT_DS4_FLASH_AOT_PREFILL_REGIMES)
    list(APPEND DS4RT_DS4_PRO_AOT_OBJECTS
      "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_prefill_m${regime}_topk6.o")
    list(APPEND DS4RT_DS4_PRO_AOT_HEADERS
      "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_prefill_m${regime}_topk6.h")
  endforeach()
  add_custom_command(
    OUTPUT
      "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_spark_aot.stamp"
      "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_spark_moe_aot.meta"
      ${DS4RT_DS4_PRO_AOT_OBJECTS}
      ${DS4RT_DS4_PRO_AOT_HEADERS}
    COMMAND ${DS4RT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${DS4RT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_ds4_flash_spark_moe_aot.py"
      --output-dir "${DS4RT_DS4_PRO_AOT_DIR}"
      --target-sms 48
      --model-variant pro
    COMMAND "${CMAKE_COMMAND}" -E touch
      "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_spark_aot.stamp"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_ds4_flash_spark_moe_aot.py"
      ${DS4RT_SPARKINFER_PROVENANCE_INPUTS}
      ${DS4RT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting DeepSeek-V4-Pro EXL3-only Spark expert-TP4 AOT kernels"
    VERBATIM
  )
  add_custom_target(ds4rt_ds4_pro_aot_export
    DEPENDS
      "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_spark_aot.stamp"
      "${DS4RT_DS4_PRO_AOT_DIR}/ds4_pro_spark_moe_aot.meta"
      ${DS4RT_DS4_PRO_AOT_OBJECTS}
      ${DS4RT_DS4_PRO_AOT_HEADERS}
  )
  add_dependencies(ds4rt_ds4_pro_aot_export ds4rt_verify_sparkinfer_source)

  set_source_files_properties(${DS4RT_DS4_FLASH_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  set_source_files_properties(${DS4RT_DS4_PRO_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND DS4RT_NATIVE_SOURCES
    cuda/kernels/ds4_flash_aot.cu
    cuda/kernels/ds4_pro_aot.cu
    ${DS4RT_DS4_FLASH_AOT_OBJECTS}
    ${DS4RT_DS4_PRO_AOT_OBJECTS}
  )
endif()

if(DS4RT_ENABLE_SPARKINFER_COORDINATOR_AOT)
  if(NOT DS4RT_ENABLE_CUDA)
    message(FATAL_ERROR
      "DS4RT_ENABLE_SPARKINFER_COORDINATOR_AOT=ON requires DS4RT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(DS4RT_B12X_COORDINATOR_AOT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/b12x_coordinator_aot"
  )
  set(DS4RT_B12X_COORDINATOR_AOT_OBJECTS
    "${DS4RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m8.o"
    "${DS4RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m16_candidate.o"
    "${DS4RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1.o"
    "${DS4RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m16_candidate.o"
    "${DS4RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1_tn64_candidate.o"
  )
  set(DS4RT_B12X_COORDINATOR_AOT_HEADERS
    "${DS4RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m8.h"
    "${DS4RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m16_candidate.h"
    "${DS4RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1.h"
    "${DS4RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m16_candidate.h"
    "${DS4RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1_tn64_candidate.h"
    "${DS4RT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot_config.h"
  )
  add_custom_command(
    OUTPUT
      "${DS4RT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.stamp"
      "${DS4RT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.meta"
      ${DS4RT_B12X_COORDINATOR_AOT_OBJECTS}
      ${DS4RT_B12X_COORDINATOR_AOT_HEADERS}
    COMMAND ${DS4RT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${DS4RT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_coordinator_aot.py"
      --output-dir "${DS4RT_B12X_COORDINATOR_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_coordinator_aot.py"
      ${DS4RT_SPARKINFER_PROVENANCE_INPUTS}
      ${DS4RT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting coordinator SparkInfer AOT kernels"
    VERBATIM
  )
  add_custom_target(ds4rt_b12x_coordinator_aot_export
    DEPENDS
      "${DS4RT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.stamp"
      "${DS4RT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.meta"
      ${DS4RT_B12X_COORDINATOR_AOT_OBJECTS}
      ${DS4RT_B12X_COORDINATOR_AOT_HEADERS}
  )
  add_dependencies(
    ds4rt_b12x_coordinator_aot_export
    ds4rt_verify_sparkinfer_source
  )
  set_source_files_properties(${DS4RT_B12X_COORDINATOR_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND DS4RT_NATIVE_SOURCES
    cuda/kernels/b12x_coordinator.cu
    ${DS4RT_B12X_COORDINATOR_AOT_OBJECTS}
  )
endif()

if(DS4RT_ENABLE_W8A16_AOT)
  if(NOT DS4RT_ENABLE_CUDA)
    message(FATAL_ERROR "DS4RT_ENABLE_W8A16_AOT=ON requires DS4RT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(DS4RT_W8A16_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/w8a16_aot")
  set(DS4RT_W8A16_AOT_HEADER
    "${DS4RT_W8A16_AOT_DIR}/w8a16_row_major_aot.h"
  )
  set(DS4RT_W8A16_AOT_SOURCE
    "${DS4RT_W8A16_AOT_DIR}/w8a16_row_major_aot.cc"
  )
  set(DS4RT_W8A16_PACKED_O_AOT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/w8a16_packed_o_aot"
  )
  set(DS4RT_W8A16_PACKED_O_AOT_OBJECTS
    "${DS4RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m16.o"
    "${DS4RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m32.o"
    "${DS4RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m64.o"
    "${DS4RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m128.o"
    "${DS4RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m256.o"
  )
  set(DS4RT_W8A16_PACKED_O_AOT_HEADERS
    "${DS4RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m16.h"
    "${DS4RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m32.h"
    "${DS4RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m64.h"
    "${DS4RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m128.h"
    "${DS4RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m256.h"
    "${DS4RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot_config.h"
  )
  add_custom_command(
    OUTPUT
      "${DS4RT_W8A16_AOT_DIR}/w8a16_row_major_aot.stamp"
      "${DS4RT_W8A16_AOT_HEADER}"
      "${DS4RT_W8A16_AOT_SOURCE}"
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_row_major_aot.py"
      --output-dir "${DS4RT_W8A16_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_row_major_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/tune_w8a16_triton_prefill.py"
    COMMENT "Exporting coordinator row-major W8A16 AOT kernels"
    VERBATIM
  )
  add_custom_target(ds4rt_w8a16_aot_export
    DEPENDS
      "${DS4RT_W8A16_AOT_DIR}/w8a16_row_major_aot.stamp"
      "${DS4RT_W8A16_AOT_HEADER}"
      "${DS4RT_W8A16_AOT_SOURCE}"
  )
  add_custom_command(
    OUTPUT
      "${DS4RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.stamp"
      "${DS4RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.meta"
      ${DS4RT_W8A16_PACKED_O_AOT_OBJECTS}
      ${DS4RT_W8A16_PACKED_O_AOT_HEADERS}
    COMMAND ${DS4RT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${DS4RT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_packed_o_aot.py"
      --output-dir "${DS4RT_W8A16_PACKED_O_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_packed_o_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/tune_w8a16_cute_packed_prefill.py"
      ${DS4RT_SPARKINFER_PROVENANCE_INPUTS}
      ${DS4RT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting coordinator packed W8A16 O-projection AOT kernels"
    VERBATIM
  )
  add_custom_target(ds4rt_w8a16_packed_o_aot_export
    DEPENDS
      "${DS4RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.stamp"
      "${DS4RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.meta"
      ${DS4RT_W8A16_PACKED_O_AOT_OBJECTS}
      ${DS4RT_W8A16_PACKED_O_AOT_HEADERS}
  )
  add_dependencies(
    ds4rt_w8a16_packed_o_aot_export
    ds4rt_verify_sparkinfer_source
  )
  set_source_files_properties("${DS4RT_W8A16_AOT_SOURCE}" PROPERTIES
    GENERATED TRUE
  )
  set_source_files_properties(${DS4RT_W8A16_PACKED_O_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND DS4RT_NATIVE_SOURCES
    "${DS4RT_W8A16_AOT_SOURCE}"
    cuda/kernels/w8a16_packed.cu
    ${DS4RT_W8A16_PACKED_O_AOT_OBJECTS}
  )
endif()

if(DS4RT_ENABLE_SPARKINFER_AOT OR DS4RT_ENABLE_DS4_FLASH_AOT OR
   DS4RT_ENABLE_SPARKINFER_COORDINATOR_AOT OR DS4RT_ENABLE_W8A16_AOT)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m cutlass.cute.export.aot_config --libdir
    OUTPUT_VARIABLE DS4RT_B12X_AOT_RUNTIME_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  find_library(DS4RT_B12X_AOT_RUNTIME_LIBRARY
    NAMES cute_dsl_runtime
    PATHS "${DS4RT_B12X_AOT_RUNTIME_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
  )
endif()

add_library(ds4rt_native SHARED ${DS4RT_NATIVE_SOURCES})
target_include_directories(ds4rt_native PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(DS4RT_ENABLE_XGRAMMAR)
  target_link_libraries(ds4rt_native PRIVATE ds4rt_xgrammar)
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_XGRAMMAR=1)
else()
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_XGRAMMAR=0)
endif()
if(DS4RT_ENABLE_CUDA)
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_CUDA=1)
  target_link_libraries(ds4rt_native PUBLIC CUDA::cudart CUDA::cublas CUDA::cuda_driver)
  set_target_properties(ds4rt_native PROPERTIES
    CUDA_ARCHITECTURES "${DS4RT_CUDA_ARCHITECTURES}"
  )
else()
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_CUDA=0)
endif()
if(DS4RT_BUILD_FLASHINFER_PACKED_MLA)
  target_include_directories(ds4rt_native PRIVATE "${DS4RT_FLASHINFER_INCLUDE_DIR}")
endif()
if(DS4RT_ENABLE_SPARKINFER_AOT)
  add_dependencies(
    ds4rt_native
    ds4rt_b12x_aot_export
    ds4rt_b12x_m1_parity_aot_export
  )
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_B12X_AOT=1)
  target_include_directories(ds4rt_native PRIVATE "${DS4RT_B12X_AOT_DIR}")
else()
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_B12X_AOT=0)
endif()
if(DS4RT_ENABLE_DS4_FLASH_AOT)
  add_dependencies(
    ds4rt_native
    ds4rt_ds4_flash_aot_export
    ds4rt_ds4_pro_aot_export
  )
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_DS4_FLASH_AOT=1)
  target_include_directories(ds4rt_native PRIVATE "${DS4RT_DS4_FLASH_AOT_DIR}")
  target_include_directories(ds4rt_native PRIVATE "${DS4RT_DS4_PRO_AOT_DIR}")
else()
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_DS4_FLASH_AOT=0)
endif()
if(DS4RT_ENABLE_SPARKINFER_COORDINATOR_AOT)
  add_dependencies(ds4rt_native ds4rt_b12x_coordinator_aot_export)
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_B12X_COORDINATOR_AOT=1)
  target_include_directories(ds4rt_native PRIVATE "${DS4RT_B12X_COORDINATOR_AOT_DIR}")
else()
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_B12X_COORDINATOR_AOT=0)
endif()
if(DS4RT_ENABLE_W8A16_AOT)
  add_dependencies(ds4rt_native ds4rt_w8a16_aot_export)
  add_dependencies(ds4rt_native ds4rt_w8a16_packed_o_aot_export)
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_W8A16_AOT=1)
  target_include_directories(ds4rt_native PRIVATE "${DS4RT_W8A16_AOT_DIR}")
  target_include_directories(ds4rt_native PRIVATE "${DS4RT_W8A16_PACKED_O_AOT_DIR}")
else()
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_W8A16_AOT=0)
endif()
if(DS4RT_ENABLE_SPARKINFER_AOT OR DS4RT_ENABLE_DS4_FLASH_AOT OR
   DS4RT_ENABLE_SPARKINFER_COORDINATOR_AOT OR DS4RT_ENABLE_W8A16_AOT)
  target_link_libraries(ds4rt_native PUBLIC "${DS4RT_B12X_AOT_RUNTIME_LIBRARY}")
  set_property(TARGET ds4rt_native APPEND PROPERTY BUILD_RPATH "${DS4RT_B12X_AOT_RUNTIME_DIR}")
endif()
if(DS4RT_ENABLE_RDMA)
  find_path(DS4RT_IBVERBS_INCLUDE_DIR infiniband/verbs.h)
  find_library(DS4RT_IBVERBS_LIBRARY NAMES ibverbs)
  if(NOT DS4RT_IBVERBS_INCLUDE_DIR OR NOT DS4RT_IBVERBS_LIBRARY)
    message(FATAL_ERROR "DS4RT_ENABLE_RDMA=ON requires libibverbs headers and library")
  endif()
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_RDMA=1)
  target_include_directories(ds4rt_native PRIVATE ${DS4RT_IBVERBS_INCLUDE_DIR})
  target_link_libraries(ds4rt_native PUBLIC ${DS4RT_IBVERBS_LIBRARY})
else()
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_RDMA=0)
endif()
if(DS4RT_ENABLE_NCCL)
  if(NOT DS4RT_ENABLE_CUDA)
    message(FATAL_ERROR "DS4RT_ENABLE_NCCL=ON requires DS4RT_ENABLE_CUDA=ON")
  endif()
  find_path(DS4RT_NCCL_INCLUDE_DIR nccl.h REQUIRED)
  find_library(DS4RT_NCCL_LIBRARY NAMES nccl REQUIRED)
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_NCCL=1)
  target_include_directories(ds4rt_native PRIVATE ${DS4RT_NCCL_INCLUDE_DIR})
  target_link_libraries(ds4rt_native PUBLIC ${DS4RT_NCCL_LIBRARY})
else()
  target_compile_definitions(ds4rt_native PRIVATE DS4RT_NATIVE_ENABLE_NCCL=0)
endif()

add_executable(ds4rt_native_selftest tests/native_selftest.cc)
target_link_libraries(ds4rt_native_selftest PRIVATE ds4rt_native)
# The self-test intentionally uses assertions for both the ABI call and its
# result check. Keep those expressions live in Release builds; defining
# NDEBUG would otherwise skip allocations and make the following copy probe
# dereference an uninitialized buffer.
if(MSVC)
  target_compile_options(ds4rt_native_selftest PRIVATE /UNDEBUG)
else()
  target_compile_options(ds4rt_native_selftest PRIVATE -UNDEBUG)
endif()

enable_testing()
add_test(NAME ds4rt_native_selftest COMMAND ds4rt_native_selftest)

if(DS4RT_ENABLE_XGRAMMAR)
  add_executable(ds4rt_xgrammar_selftest tests/xgrammar_selftest.cc)
  target_link_libraries(ds4rt_xgrammar_selftest PRIVATE ds4rt_native)
  target_compile_definitions(ds4rt_xgrammar_selftest PRIVATE
    "DS4RT_XGRAMMAR_TEST_TOKENIZER=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/xgrammar_tiny_tokenizer.json\""
  )
  if(MSVC)
    target_compile_options(ds4rt_xgrammar_selftest PRIVATE /UNDEBUG)
  else()
    target_compile_options(ds4rt_xgrammar_selftest PRIVATE -UNDEBUG)
  endif()
  add_test(NAME ds4rt_xgrammar_selftest COMMAND ds4rt_xgrammar_selftest)
endif()

if(DS4RT_ENABLE_CUDA)
  add_executable(ds4rt_cuda_selftest tests/cuda_selftest.cc)
  target_link_libraries(ds4rt_cuda_selftest PRIVATE ds4rt_native CUDA::cudart CUDA::cublas CUDA::cuda_driver)
  add_test(NAME ds4rt_cuda_selftest COMMAND ds4rt_cuda_selftest)
endif()

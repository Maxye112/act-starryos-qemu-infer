cmake_minimum_required(VERSION 3.16)
project(act_ort_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime package root")
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()

add_executable(act_ort_infer
  src/act_ort_infer.cpp
)

target_include_directories(act_ort_infer PRIVATE
  ${ONNXRUNTIME_ROOT}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

target_link_directories(act_ort_infer PRIVATE
  ${ONNXRUNTIME_ROOT}/lib
)

target_link_libraries(act_ort_infer PRIVATE onnxruntime)

target_compile_options(act_ort_infer PRIVATE
  -O3
  -ffast-math
)

set_target_properties(act_ort_infer PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

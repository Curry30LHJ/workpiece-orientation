cmake_minimum_required(VERSION 3.20)
project(ppshitu_rec_benchmark LANGUAGES CXX)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED PADDLE_INFERENCE_ROOT)
  message(FATAL_ERROR "PADDLE_INFERENCE_ROOT is required")
endif()
if(NOT DEFINED OPENCV_ROOT)
  message(FATAL_ERROR "OPENCV_ROOT is required")
endif()

set(PADDLE_INFERENCE_HEADER
    "${PADDLE_INFERENCE_ROOT}/paddle/include/paddle_inference_api.h")
set(PADDLE_INFERENCE_LIBRARY
    "${PADDLE_INFERENCE_ROOT}/paddle/lib/paddle_inference.lib")
if(NOT EXISTS "${PADDLE_INFERENCE_HEADER}")
  message(FATAL_ERROR "Missing Paddle header: ${PADDLE_INFERENCE_HEADER}")
endif()
if(NOT EXISTS "${PADDLE_INFERENCE_LIBRARY}")
  message(FATAL_ERROR "Missing Paddle library: ${PADDLE_INFERENCE_LIBRARY}")
endif()

find_package(OpenCV REQUIRED PATHS
    "${OPENCV_ROOT}"
    "${OPENCV_ROOT}/build"
    NO_DEFAULT_PATH)

execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    OUTPUT_VARIABLE WORKPIECE_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT WORKPIECE_GIT_COMMIT)
  set(WORKPIECE_GIT_COMMIT "unknown")
endif()

add_library(ppshitu_protocol STATIC
    src/ppshitu_protocol.cpp)
target_include_directories(ppshitu_protocol PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(ppshitu_protocol_probe
    src/protocol_probe_main.cpp)
target_link_libraries(ppshitu_protocol_probe PRIVATE
    ppshitu_protocol)

add_executable(ppshitu_rec_benchmark
    src/main.cpp
    src/preprocess.cpp
    src/feature_extractor.cpp)
target_include_directories(ppshitu_rec_benchmark PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(ppshitu_rec_benchmark SYSTEM PRIVATE
    "${PADDLE_INFERENCE_ROOT}/paddle/include"
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ppshitu_rec_benchmark PRIVATE
    "${PADDLE_INFERENCE_LIBRARY}"
    ${OpenCV_LIBS}
    Psapi
    Bcrypt)

target_compile_definitions(ppshitu_rec_benchmark PRIVATE
    WORKPIECE_GIT_COMMIT="${WORKPIECE_GIT_COMMIT}")

if(MSVC)
  target_compile_options(ppshitu_protocol PRIVATE
      /utf-8 /W4 /permissive-)
  target_compile_options(ppshitu_protocol_probe PRIVATE
      /utf-8 /W4 /permissive-)
  target_compile_options(ppshitu_rec_benchmark PRIVATE
      /utf-8 /W4 /permissive- /external:anglebrackets /external:W0)
endif()

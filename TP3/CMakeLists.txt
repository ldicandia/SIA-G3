cmake_minimum_required(VERSION 3.16)
project(tp3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-Wall -Wextra)

# Engine: explicit source list so new modules (e.g. core/mlp.cpp) are added deliberately.
add_library(tp3_core STATIC
  core/matrix.cpp
  core/activations.cpp
  core/training.cpp
  core/perceptron.cpp
  core/mlp.cpp
  core/validation/datasets.cpp
  core/io/run_json.cpp
)
target_include_directories(tp3_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)

# CLI binary.
add_executable(tp3 core/main.cpp)
target_link_libraries(tp3 PRIVATE tp3_core)

# doctest runner: globbed so new test files register automatically.
file(GLOB TP3_TEST_SOURCES CONFIGURE_DEPENDS core/tests/*.cpp)
add_executable(tp3_tests ${TP3_TEST_SOURCES})
target_link_libraries(tp3_tests PRIVATE tp3_core)
target_include_directories(tp3_tests SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

enable_testing()
add_test(NAME tp3_tests COMMAND tp3_tests)

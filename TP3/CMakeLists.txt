cmake_minimum_required(VERSION 3.16)
project(tp3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-Wall -Wextra)

# Engine: globbed recursively so all core components are automatically included
file(GLOB_RECURSE TP3_CORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/core/src/*.cpp)
list(REMOVE_ITEM TP3_CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/core/src/main.cpp)
add_library(tp3_core STATIC ${TP3_CORE_SOURCES})
target_include_directories(tp3_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/include)

# CLI binary.
add_executable(tp3 core/src/main.cpp)
target_link_libraries(tp3 PRIVATE tp3_core)

# doctest runner: globbed so new test files register automatically.
file(GLOB TP3_TEST_SOURCES CONFIGURE_DEPENDS core/tests/*.cpp)
add_executable(tp3_tests ${TP3_TEST_SOURCES})
target_link_libraries(tp3_tests PRIVATE tp3_core)
target_include_directories(tp3_tests SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_compile_definitions(tp3_tests PRIVATE "TP3_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")

enable_testing()
add_test(NAME tp3_tests COMMAND tp3_tests)

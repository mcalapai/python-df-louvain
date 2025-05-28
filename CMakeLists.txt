cmake_minimum_required(VERSION 3.18)
project(df_louvain_cpp_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 REQUIRED)
find_package(OpenMP)

pybind11_add_module(df_louvain_cpp src/wrapper.cpp)

target_include_directories(df_louvain_cpp PRIVATE
    ${pybind11_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp_src/inc"
)

# Add compile definitions
target_compile_definitions(df_louvain_cpp PRIVATE OPENMP=1 BUILD=0) # Set OPENMP and BUILD define

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling OpenMP support.")
    target_compile_options(df_louvain_cpp PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(df_louvain_cpp PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found. Building without OpenMP support.")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(df_louvain_cpp PRIVATE -O3 -Wall -Wextra)
elseif(MSVC)
    target_compile_options(df_louvain_cpp PRIVATE /O2 /W3)
endif()

install(TARGETS df_louvain_cpp
        LIBRARY DESTINATION df_louvain
        RUNTIME DESTINATION df_louvain
       )
cmake_minimum_required(VERSION 3.18)
project(viteo LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)

# Find Python and nanobind
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Try to find nanobind
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
  RESULT_VARIABLE NB_RESULT)

if(NB_RESULT)
  message(FATAL_ERROR "nanobind not found. Install with: pip install nanobind")
endif()

list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Find MLX
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import mlx; print(mlx.__path__[0])"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE MLX_PYTHON_DIR
  RESULT_VARIABLE MLX_RESULT)

if(MLX_RESULT)
  message(FATAL_ERROR "MLX not found. Install with: pip install mlx")
endif()

# MLX include and library paths
set(MLX_INCLUDE_DIR "${MLX_PYTHON_DIR}/include")
set(MLX_LIB_DIR "${MLX_PYTHON_DIR}/lib")

# Find MLX library
find_library(MLX_LIBRARY
  NAMES mlx
  PATHS "${MLX_LIB_DIR}"
  NO_DEFAULT_PATH
  REQUIRED)

message(STATUS "MLX include dir: ${MLX_INCLUDE_DIR}")
message(STATUS "MLX library: ${MLX_LIBRARY}")

# Apple frameworks
find_library(AVFOUNDATION_FRAMEWORK AVFoundation REQUIRED)
find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
find_library(COREMEDIA_FRAMEWORK CoreMedia REQUIRED)
find_library(VIDEOTOOLBOX_FRAMEWORK VideoToolbox REQUIRED)
find_library(COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(COREIMAGE_FRAMEWORK CoreImage REQUIRED)

# Source files
set(SOURCES
  src/native/src/frame_extractor.mm
  src/native/src/bindings.cpp
)

# Create Python extension module
nanobind_add_module(
  _viteo
  ${SOURCES}
)

target_include_directories(_viteo PRIVATE
  src/native/include
  ${MLX_INCLUDE_DIR}
)

target_link_libraries(_viteo PRIVATE
  ${MLX_LIBRARY}
  ${AVFOUNDATION_FRAMEWORK}
  ${COREVIDEO_FRAMEWORK}
  ${COREMEDIA_FRAMEWORK}
  ${VIDEOTOOLBOX_FRAMEWORK}
  ${COREGRAPHICS_FRAMEWORK}
  ${FOUNDATION_FRAMEWORK}
  ${COREIMAGE_FRAMEWORK}
)

# Set rpath to find MLX library at runtime
# Use @loader_path to find MLX relative to the module location
set_target_properties(_viteo PROPERTIES
  BUILD_RPATH "${MLX_LIB_DIR}"
  INSTALL_RPATH "@loader_path/mlx/lib"
  BUILD_WITH_INSTALL_RPATH FALSE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Enable Objective-C++ for .mm files
set_source_files_properties(
  src/native/src/frame_extractor.mm
  PROPERTIES
  COMPILE_FLAGS "-fobjc-arc"
)

# Install the extension module to the correct location
install(TARGETS _viteo LIBRARY DESTINATION .)
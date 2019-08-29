cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project (minkowski)

# TODO: check if it is the correct way in a conda env
# any workaround to enable cuda_add_library()?
find_package(CUDA 8.0 REQUIRED)

# collect include paths
execute_process(COMMAND bash -c
  "python -c 'from distutils.sysconfig import get_python_inc; \
   print(get_python_inc(), end=\"\")'"
  OUTPUT_VARIABLE PYTHON_INCLUDES)

get_filename_component(PY3_MINK_INCLUDES ${PYTHON_INCLUDES} DIRECTORY)

execute_process(COMMAND bash -c
  "python -c \
  'from torch.utils.cpp_extension import include_paths; \
   [print(p, end=\";\") for p in include_paths()]'"
  OUTPUT_VARIABLE PYTORCH_INCLUDES)

set(INCLUDE_DIRS "")
list(APPEND INCLUDE_DIRS
  ${PYTHON_INCLUDES}
  ${PY3_MINK_INCLUDES}
  ${PYTORCH_INCLUDES})

foreach(include ${INCLUDE_DIRS})
  message(${include})
endforeach()
include_directories(${INCLUDE_DIRS})

# collect source files
file(GLOB_RECURSE ALL_CPP_SOURCE_FILES "src/*.cpp")
file(GLOB_RECURSE ALL_CUDA_SOURCE_FILES "src/*.cu")

# flags for cuda code generation
set(CUDA_NVCC_FLAGS
  -gencode arch=compute_30,code=sm_30
	-gencode arch=compute_35,code=sm_35
	-gencode arch=compute_50,code=sm_50
	-gencode arch=compute_52,code=sm_52
	-gencode arch=compute_60,code=sm_60
	-gencode arch=compute_61,code=sm_61
	-gencode arch=compute_61,code=compute_61
  --expt-relaxed-constexpr)

# build
# TODO: correctly link libraries
add_library(${PROJECT_NAME} STATIC ${ALL_CPP_SOURCE_FILES})
cuda_add_library(MinkCuda STATIC ${ALL_CUDA_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} MinkCuda)

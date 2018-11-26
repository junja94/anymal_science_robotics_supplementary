
cmake_minimum_required(VERSION 3.5)
project(anymal_raisim)

#==
# Dependencies
#==

find_package(RAI CONFIG REQUIRED)
if (RAI_FOUND)
  message(STATUS "rlAPP: Found RAI!")
  list(APPEND ${PROJECT_NAME}_INCLUDE_DIRS ${RAI_INCLUDE_DIRS})
  list(APPEND ${PROJECT_NAME}_LIBRARIES ${RAI_LIBRARIES})
  list(APPEND ${PROJECT_NAME}_QUAD_LIBRARIES ${RAI_LIBRARIES})
endif()

# Search for raiSim
find_package(raiSim CONFIG REQUIRED)
if (raiSim_FOUND)
    message(STATUS "Examples: Found raiSim!")
    list(APPEND ${PROJECT_NAME}_INCLUDE_DIRS ${raiSim_INCLUDE_DIRS})
    list(APPEND ${PROJECT_NAME}_LIBRARIES ${raiSim_LIBRARIES})
endif()

# GLog and GFlags
list(APPEND ${PROJECT_NAME}_LIBRARIES glog gflags stdc++fs)

# Simplify by removing duplicate paths
list(REMOVE_DUPLICATES ${PROJECT_NAME}_INCLUDE_DIRS)
list(REMOVE_DUPLICATES ${PROJECT_NAME}_LIBRARIES)

#==
# Build configuration: for use via add_subdirectory()
#==

# Set C++ language version
set(CMAKE_CXX_STANDARD 14)

# Set GCC version
set(CMAKE_CXX_COMPILER "/usr/bin/g++-6")

# Set GCC flags
set(CMAKE_CXX_FLAGS "-Wl,--no-as-needed -fopenmp -O3 -w -funroll-loops")

# Set include directories
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
        ${CMAKE_CURRENT_SOURCE_DIR}/functions
        ${CMAKE_CURRENT_SOURCE_DIR}/memory
        ${CMAKE_CURRENT_SOURCE_DIR}/task/include
        ${${PROJECT_NAME}_INCLUDE_DIRS}
)


### apps for self-righting
add_executable(anymal_self_righting
        applications/self_righting/anymal_self_righting.cpp)
target_link_libraries(anymal_self_righting
        ${${PROJECT_NAME}_LIBRARIES} ${RAI_GRAPHICS_OPENGL_LINK} ${raiCommon_LIBRARIES} ${RAI_SIM_LINK})

add_executable(test_self_righting
        applications/self_righting/test_self_righting.cpp)
target_link_libraries(test_self_righting
        ${${PROJECT_NAME}_LIBRARIES} ${RAI_GRAPHICS_OPENGL_LINK} ${raiCommon_LIBRARIES} ${RAI_SIM_LINK})

### apps for walking
add_executable(test_walk
        applications/walk/test_walk.cpp)
target_link_libraries(test_walk
        ${${PROJECT_NAME}_LIBRARIES} ${RAI_GRAPHICS_OPENGL_LINK} ${raiCommon_LIBRARIES} ${RAI_SIM_LINK})

add_executable(anymal_walk
        applications/walk/anymal_walk.cpp)
target_link_libraries(anymal_walk
        ${${PROJECT_NAME}_LIBRARIES} ${RAI_GRAPHICS_OPENGL_LINK} ${raiCommon_LIBRARIES} ${RAI_SIM_LINK})
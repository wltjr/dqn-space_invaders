################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 3.20)
project(q-space_invaders)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-g -Wall -Wextra -Wpedantic)
endif()


################################################################################
# Find packages and libraries dependencies
################################################################################
find_package(ale REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)


################################################################################
# Build
################################################################################
include_directories(src ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

# executable binaries - DQN Space Invaders
add_executable(dqnsi src/main.cpp
                     src/replay_memory.cpp)

# link executable binaries to shared object libraries
target_link_libraries(dqnsi ale::ale-lib ${OpenCV_LIBS} ${TORCH_LIBRARIES})

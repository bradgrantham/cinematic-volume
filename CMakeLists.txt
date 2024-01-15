cmake_minimum_required(VERSION 3.12.2)

project(volume)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

find_package(Vulkan REQUIRED)

find_package(glfw3 3.3 CONFIG REQUIRED)

if(NOT Vulkan_FOUND)
    message(
        FATAL_ERROR
            "Vulkan package was not found"
    )
endif()

add_executable(volume main.cpp vectormath.cpp manipulator.cpp)
target_link_libraries(volume glfw)
target_link_libraries(volume ${Vulkan_LIBRARIES})
target_include_directories(volume PRIVATE ${Vulkan_INCLUDE_DIRS})
# target_include_directories(volume PRIVATE ${GLFW_INCLUDE_DIR})
set_property(TARGET volume PROPERTY CXX_STANDARD 20)

if(MSVC)
  target_compile_options(volume PRIVATE /W4 /WX)
else()
  target_compile_options(volume PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-missing-field-initializers)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(volume Threads::Threads)

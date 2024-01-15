cmake_minimum_required(VERSION 3.16)

project(ct-view)
 
# brad - ???
# specify that we have a config file. This is required, since DCMTK package for Ubuntu 14.04 
# uses /usr/include/dcmtk/config/cfunix.h to setup dcmtk. The file is only loaded by
# DCMTK's /usr/include/dcmtk/config/osconfig.h when HAVE_CONFIG_H is set.
# add_definitions(-DHAVE_CONFIG_H)

# define location of the source code. This is used so that we can refer in the example
# code to the DCMIMAGES folder relatively to the source folder.
# add_definitions(-DSOURCE_CODE_LOCATION="${CMAKE_SOURCE_DIR}")

add_executable(ct-view ct-view.cpp)

find_package(DCMTK REQUIRED)

target_include_directories(ct-view PRIVATE ${DCMTK_INCLUDE_DIRS})

target_link_libraries(ct-view ${DCMTK_LIBRARIES})

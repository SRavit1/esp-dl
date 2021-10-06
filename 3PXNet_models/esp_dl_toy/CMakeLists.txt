cmake_minimum_required(VERSION 3.5)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# -O3 -std=c99 -lm -g -ffloat-store -mpopcnt  -DPCK32 -DARCH64 -DGCC -DPCNTSW
idf_build_set_property(COMPILE_OPTIONS -O0 APPEND)
idf_build_set_property(COMPILE_OPTIONS -g APPEND)

project(toy)

cmake_minimum_required(VERSION 3.5)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# -O3 -std=c99 -lm -g -ffloat-store -mpopcnt  -DPCK32 -DARCH64 -DGCC -DPCNTSW
idf_build_set_property(COMPILE_OPTIONS -O0 APPEND)
idf_build_set_property(COMPILE_OPTIONS -std=c99 APPEND)
idf_build_set_property(COMPILE_OPTIONS -lm APPEND)
idf_build_set_property(COMPILE_OPTIONS -g APPEND)
idf_build_set_property(COMPILE_OPTIONS -ffloat-store APPEND)
#idf_build_set_property(COMPILE_OPTIONS -mpopcnt APPEND)
idf_build_set_property(COMPILE_OPTIONS -DPCK32 APPEND)
idf_build_set_property(COMPILE_OPTIONS -DARCH32 APPEND)
idf_build_set_property(COMPILE_OPTIONS -DGCC APPEND)
#idf_build_set_property(COMPILE_OPTIONS -DPCNTSW APPEND)

idf_build_set_property(COMPILE_OPTIONS -Wno-return-type APPEND)

project(custom_model)

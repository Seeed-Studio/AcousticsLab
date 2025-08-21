cmake_minimum_required(VERSION 3.12.4)

set(PROJ_INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}
)

file(GLOB_RECURSE PROJ_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
)

set(ACOUSTICS_PORTING_LIB_TFLM_ENABLE ON CACHE BOOL "Enable TFLM support in acoustics-porting")
set(ACOUSTICS_PORTING_LIB_DLFFT_ENABLE ON CACHE BOOL "Enable dl-fft support in acoustics-porting")

idf_component_register(
    SRCS ${PROJ_SRCS}
    INCLUDE_DIRS ${PROJ_INCLUDE_DIRS}
    REQUIRES freertos esp_psram esp_system acoustics-porting
)

add_compile_options(-fdiagnostics-color=always -ffast-math -O2)

cmake_minimum_required(VERSION 3.12.4)

message(STATUS "Configuring AcousticsLab Porting SDK")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ACOUSTICS_SDK_VERSION_MAJOR 1)
set(ACOUSTICS_SDK_VERSION_MINOR 7)
set(ACOUSTICS_SDK_VERSION_PATCH 24)

set(ACOUSTICS_PORTING_INCLUDES_DIR
    ${CMAKE_CURRENT_LIST_DIR}/porting
)
message(STATUS "Include directory: ${ACOUSTICS_PORTING_INCLUDES_DIR}")


file(GLOB_RECURSE ACOUSTICS_PORTING_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/porting/*.h
    ${CMAKE_CURRENT_LIST_DIR}/porting/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/porting/*.cpp
)
message(STATUS "ACOUSTICS_PORTING_SRCS: ${ACOUSTICS_PORTING_SRCS}")

set_property(GLOBAL PROPERTY ACOUSTICS_SDK_TARGET "POSIX")


set(ACOUSTICS_INCLUDES_DIR
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/
)

file(GLOB_RECURSE ACOUSTICS_SRCS_API
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/api/*.c
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/api/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/api/*.cpp
)

file(GLOB_RECURSE ACOUSTICS_SRCS_CORE
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/core/*.c
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/core/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/core/*.cpp
)

file(GLOB_RECURSE ACOUSTICS_SRCS_HAL
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/hal/*.c
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/hal/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/hal/*.cpp
)

file(GLOB_RECURSE ACOUSTICS_SRCS_ALGORITHMS
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/algorithms/*.c
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/algorithms/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/algorithms/*.cpp
)

file(GLOB_RECURSE ACOUSTICS_SRCS_MODULE
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/module/*.c
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/module/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/module/*.cpp
)
message(STATUS "ACOUSTICS_SRCS_MODULE: ${ACOUSTICS_SRCS_MODULE}")

set(ACOUSTICS_SRCS
    ${ACOUSTICS_SRCS_API}
    ${ACOUSTICS_SRCS_CORE}
    ${ACOUSTICS_SRCS_HAL}
    ${ACOUSTICS_SRCS_ALGORITHMS}
    ${ACOUSTICS_SRCS_MODULE}
)

list(APPEND ACOUSTICS_PORTING_SRCS ${ACOUSTICS_SRCS})
list(APPEND ACOUSTICS_PORTING_INCLUDES_DIR ${ACOUSTICS_INCLUDES_DIR})


set(ACOUSTICS_REQUIRES
    driver
    freertos
    efuse
    spi_flash
    esp_timer
    esp_psram
    esp_partition
)

set(ACOUSTICS_PRIV_REQUIRES
    lis3dhtr
    littlefs
)

if(NOT DEFINED ACOUSTICS_PORTING_LIB_TFLM_ENABLE)
    set(PORTING_LIB_TFLM_ENABLE 0)
    message(STATUS "TFLM support is disabled by default")
else()
    message(STATUS "TFLM support is set to: ${ACOUSTICS_PORTING_LIB_TFLM_ENABLE}")
    if(ACOUSTICS_PORTING_LIB_TFLM_ENABLE STREQUAL "1" OR ACOUSTICS_PORTING_LIB_TFLM_ENABLE STREQUAL "ON")
        set(PORTING_LIB_TFLM_ENABLE 1)
        list(APPEND ACOUSTICS_REQUIRES esp-tflite-micro)
        message(STATUS "TFLM support is enabled")
    else()
        set(PORTING_LIB_TFLM_ENABLE 0)
        message(STATUS "TFLM support is disabled")
    endif()
endif()

if(NOT DEFINED ACOUSTICS_PORTING_LIB_DLFFT_ENABLE)
    set(PORTING_LIB_DLFFT_ENABLE 0)
    message(STATUS "DL_FFT support is disabled by default")
else()
    message(STATUS "DL_FFT support is set to: ${ACOUSTICS_PORTING_LIB_DLFFT_ENABLE}")
    if(ACOUSTICS_PORTING_LIB_DLFFT_ENABLE STREQUAL "1" OR ACOUSTICS_PORTING_LIB_DLFFT_ENABLE STREQUAL "ON")
        set(PORTING_LIB_DLFFT_ENABLE 1)
        list(APPEND ACOUSTICS_REQUIRES dl_fft)
        message(STATUS "DL_FFT support is enabled")
    else()
        set(PORTING_LIB_DLFFT_ENABLE 0)
        message(STATUS "DL_FFT support is disabled")
    endif()
endif()

idf_component_register(
    SRCS ${ACOUSTICS_PORTING_SRCS}
    INCLUDE_DIRS ${ACOUSTICS_PORTING_INCLUDES_DIR}
    REQUIRES freertos ${ACOUSTICS_REQUIRES}
    PRIV_REQUIRES ${ACOUSTICS_PRIV_REQUIRES}
)

add_compile_options(-fdiagnostics-color=always -ffast-math -O2)

target_compile_definitions(${COMPONENT_LIB} PUBLIC
    CORE_VERSION_MAJOR=${ACOUSTICS_SDK_VERSION_MAJOR}
    CORE_VERSION_MINOR=${ACOUSTICS_SDK_VERSION_MINOR}
    CORE_VERSION_PATCH=${ACOUSTICS_SDK_VERSION_PATCH}
    PORTING_LIB_TFLM_ENABLE=${PORTING_LIB_TFLM_ENABLE}
    PORTING_LIB_DLFFT_ENABLE=${PORTING_LIB_DLFFT_ENABLE}
)
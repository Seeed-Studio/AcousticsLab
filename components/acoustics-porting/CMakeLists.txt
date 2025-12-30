cmake_minimum_required(VERSION 3.12.4)

message(STATUS "Configuring AcousticsLab Porting SDK")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ACOUSTICS_SDK_VERSION_MAJOR 1)
set(ACOUSTICS_SDK_VERSION_MINOR 12)
set(ACOUSTICS_SDK_VERSION_PATCH 30)

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

file(GLOB ACOUSTICS_SRCS_API
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/api/*.c
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/api/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/api/*.cpp
)

if(NOT DEFINED ACOUSTICS_API_VERSION)
    message(WARNING "ACOUSTICS_API_VERSION is not defined, defaulting to 0")
    set(ACOUSTICS_API_VERSION "0")
endif()

if(${ACOUSTICS_API_VERSION} STREQUAL "0")
    file(GLOB_RECURSE ACOUSTICS_SRCS_API_V0
        ${CMAKE_CURRENT_LIST_DIR}/../acoustics/api/v0/*.c
        ${CMAKE_CURRENT_LIST_DIR}/../acoustics/api/v0/*.hpp
        ${CMAKE_CURRENT_LIST_DIR}/../acoustics/api/v0/*.cpp
    )
    list(APPEND ACOUSTICS_SRCS_API ${ACOUSTICS_SRCS_API_V0})
    message(STATUS "Using API version 0 sources")
elseif(${ACOUSTICS_API_VERSION} STREQUAL "1")
    file(GLOB_RECURSE ACOUSTICS_SRCS_API_V1
        ${CMAKE_CURRENT_LIST_DIR}/../acoustics/api/v1/*.c
        ${CMAKE_CURRENT_LIST_DIR}/../acoustics/api/v1/*.hpp
        ${CMAKE_CURRENT_LIST_DIR}/../acoustics/api/v1/*.cpp
    )
    list(APPEND ACOUSTICS_SRCS_API ${ACOUSTICS_SRCS_API_V1})
    message(STATUS "Using API version 1 sources")
endif()

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

file(GLOB_RECURSE ACOUSTICS_SRCS_ALGORITHM
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/algorithm/*.c
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/algorithm/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../acoustics/algorithm/*.cpp
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
    ${ACOUSTICS_SRCS_ALGORITHM}
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

if(NOT DEFINED ACOUSTICS_LIB_LIBSAMPLERATE_ENABLE)
    set(LIB_LIBSAMPLERATE_ENABLE 0)
    message(STATUS "SRC (aka libsamplerate) support is disabled by default")
else()
    message(STATUS "SRC (aka libsamplerate) support is set to: ${ACOUSTICS_LIB_LIBSAMPLERATE_ENABLE}")
    if(ACOUSTICS_LIB_LIBSAMPLERATE_ENABLE STREQUAL "1" OR ACOUSTICS_LIB_LIBSAMPLERATE_ENABLE STREQUAL "ON")
        set(LIB_LIBSAMPLERATE_DIR ${CMAKE_CURRENT_LIST_DIR}/../acoustics/3rdparty/libsamplerate)
        file(GLOB_RECURSE LIB_LIBSAMPLERATE_SRCS
            ${LIB_LIBSAMPLERATE_DIR}/src/*.c
            ${LIB_LIBSAMPLERATE_DIR}/src/*.h
        )
        list(APPEND ACOUSTICS_PORTING_SRCS ${LIB_LIBSAMPLERATE_SRCS})
        list(APPEND ACOUSTICS_PORTING_INCLUDES_DIR ${LIB_LIBSAMPLERATE_DIR}/include)

        set(LIB_LIBSAMPLERATE_ENABLE 1)
        message(STATUS "SRC (aka libsamplerate) support is enabled")
    else()
        set(LIB_LIBSAMPLERATE_ENABLE 0)
        message(STATUS "SRC (aka libsamplerate) support is disabled")
    endif()
endif()

if(NOT DEFINED ACOUSTICS_LIB_OPUS_ENABLE)
    set(LIB_OPUS_ENABLE 0)
    message(STATUS "OPUS support is disabled by default")
else()
    message(STATUS "OPUS support is set to: ${ACOUSTICS_LIB_OPUS_ENABLE}")
    if(ACOUSTICS_LIB_OPUS_ENABLE STREQUAL "1" OR ACOUSTICS_LIB_OPUS_ENABLE STREQUAL "ON")
        set(LIB_OPUS_DIR ${CMAKE_CURRENT_LIST_DIR}/../acoustics/3rdparty/opus)

        set(LIB_OPUS_CELT_DIR ${LIB_OPUS_DIR}/celt)
        file(GLOB LIB_OPUS_CELT_SRCS
            ${LIB_OPUS_CELT_DIR}/*.c
            ${LIB_OPUS_CELT_DIR}/*.h
        )

        set(LIB_OPUS_SILK_DIR ${LIB_OPUS_DIR}/silk)
        file(GLOB LIB_OPUS_SILK_SRCS
            ${LIB_OPUS_SILK_DIR}/*.c
            ${LIB_OPUS_SILK_DIR}/*.h
        )
        set(LIB_OPUS_SILK_FIXED_DIR ${LIB_OPUS_DIR}/silk/fixed)
        file(GLOB LIB_OPUS_SILK_FIXED_SRCS
            ${LIB_OPUS_SILK_FIXED_DIR}/*.c
            ${LIB_OPUS_SILK_FIXED_DIR}/*.h
        )

        set(LIB_OPUS_INCS ${LIB_OPUS_DIR}/include)
        file(GLOB_RECURSE LIB_OPUS_SRCS
            ${LIB_OPUS_DIR}/src/*.c
            ${LIB_OPUS_DIR}/src/*.h
        )

        list(APPEND ACOUSTICS_PORTING_SRCS
            ${LIB_OPUS_CELT_SRCS}
            ${LIB_OPUS_SILK_SRCS}
            ${LIB_OPUS_SILK_FIXED_SRCS}
            ${LIB_OPUS_SRCS}
        )
        list(APPEND ACOUSTICS_PORTING_INCLUDES_DIR
            ${LIB_OPUS_SILK_DIR}
            ${LIB_OPUS_CELT_DIR}
            ${LIB_OPUS_SILK_FIXED_DIR}
            ${LIB_OPUS_INCS}
        )

        set(LIB_OPUS_ENABLE 1)
        message(STATUS "OPUS support is enabled")
    else()
        set(LIB_OPUS_ENABLE 0)
        message(STATUS "OPUS support is disabled")
    endif()
endif()

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

if(NOT DEFINED ACOUSTICS_PORTING_LIB_DL_FFT_ENABLE)
    set(PORTING_LIB_DL_FFT_ENABLE 0)
    message(STATUS "DL_FFT support is disabled by default")
else()
    message(STATUS "DL_FFT support is set to: ${ACOUSTICS_PORTING_LIB_DL_FFT_ENABLE}")
    if(ACOUSTICS_PORTING_LIB_DL_FFT_ENABLE STREQUAL "1" OR ACOUSTICS_PORTING_LIB_DL_FFT_ENABLE STREQUAL "ON")
        set(PORTING_LIB_DL_FFT_ENABLE 1)
        list(APPEND ACOUSTICS_REQUIRES dl_fft)
        message(STATUS "DL_FFT support is enabled")
    else()
        set(PORTING_LIB_DL_FFT_ENABLE 0)
        message(STATUS "DL_FFT support is disabled")
    endif()
endif()

set(SUPPORTED_BOARDS XIAO_S3 RESPEAKER_LITE RESPEAKER_XVF3800)

foreach(BOARD_NAME ${SUPPORTED_BOARDS})
    set(KCONFIG_VAR "CONFIG_BOARD_${BOARD_NAME}")
    set(CPP_MACRO_NAME "PORTING_BOARD_MODEL_${BOARD_NAME}")
    if(${${KCONFIG_VAR}})
        set(${CPP_MACRO_NAME} 1)
        message(STATUS "Board selected: ${BOARD_NAME}. Defining ${CPP_MACRO_NAME}=1")
    else()
        set(${CPP_MACRO_NAME} 0)
    endif()
endforeach()

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
    LIB_LIBSAMPLERATE_ENABLE=${LIB_LIBSAMPLERATE_ENABLE}
    LIB_OPUS_ENABLE=${LIB_OPUS_ENABLE}
    PORTING_LIB_TFLM_ENABLE=${PORTING_LIB_TFLM_ENABLE}
    PORTING_LIB_DL_FFT_ENABLE=${PORTING_LIB_DL_FFT_ENABLE}
    PORTING_BOARD_MODEL_XIAO_S3=${PORTING_BOARD_MODEL_XIAO_S3}
    PORTING_BOARD_MODEL_RESPEAKER_LITE=${PORTING_BOARD_MODEL_RESPEAKER_LITE}
    PORTING_BOARD_MODEL_RESPEAKER_XVF3800=${PORTING_BOARD_MODEL_RESPEAKER_XVF3800}
)

if(LIB_OPUS_ENABLE)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE
        OPUS_BUILD=1
        FLOAT_APPROX=1
        NONTHREADSAFE_PSEUDOSTACK=1
        FIXED_POINT=1
    )
    target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-maybe-uninitialized)
endif()

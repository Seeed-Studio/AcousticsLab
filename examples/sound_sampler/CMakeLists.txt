cmake_minimum_required(VERSION 3.16)

set(ACOUSTICS_LIB_OPUS_ENABLE 1)

set(COMPONENTS
    driver
    cxx
    main
    bootloader
    bootloader_support
    app_trace
    esp_adc
    bus
    cmock
    xtensa
    partition_table
    log
    freertos
    hal
)

message(STATUS "COMPONENTS: ${COMPONENTS}")

add_compile_options(-fdiagnostics-color=always -ffast-math -O2 -Wdouble-promotion)

set(CMAKE_CXX_STANDARD 20)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(SoundSampler)

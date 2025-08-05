cmake_minimum_required(VERSION 3.12.4)

message(STATUS "Configuring AcousticsLab SDK")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(ACOUSTICS_INCLUDES_DIR
    ${CMAKE_CURRENT_LIST_DIR}/
)
message(STATUS "Include directory: ${ACOUSTICS_INCLUDES_DIR}")

file(GLOB ACOUSTICS_SRCS_API
    ${CMAKE_CURRENT_LIST_DIR}/api/*.c
    ${CMAKE_CURRENT_LIST_DIR}/api/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/api/*.cpp
)

if(NOT DEFINED ACOUSTICS_API_VERSION)
    message(WARNING "ACOUSTICS_API_VERSION is not defined, defaulting to 0")
    set(ACOUSTICS_API_VERSION "0")
endif()

if(${ACOUSTICS_API_VERSION} STREQUAL "0")
    file(GLOB_RECURSE ACOUSTICS_SRCS_API_V0
        ${CMAKE_CURRENT_LIST_DIR}/api/v0/*.c
        ${CMAKE_CURRENT_LIST_DIR}/api/v0/*.hpp
        ${CMAKE_CURRENT_LIST_DIR}/api/v0/*.cpp
    )
    list(APPEND ACOUSTICS_SRCS_API ${ACOUSTICS_SRCS_API_V0})
    message(STATUS "Using API version 0 sources")
endif()

if(NOT DEFINED ACOUSTICS_SRCS_API)
    message(FATAL_ERROR "ACOUSTICS_SRCS_API is not defined")
endif()
message(STATUS "ACOUSTICS_SRCS_API: ${ACOUSTICS_SRCS_API}")

file(GLOB_RECURSE ACOUSTICS_SRCS_CORE
    ${CMAKE_CURRENT_LIST_DIR}/core/*.c
    ${CMAKE_CURRENT_LIST_DIR}/core/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/*.cpp
)
message(STATUS "ACOUSTICS_SRCS_CORE: ${ACOUSTICS_SRCS_CORE}")

file(GLOB_RECURSE ACOUSTICS_SRCS_HAL
    ${CMAKE_CURRENT_LIST_DIR}/hal/*.c
    ${CMAKE_CURRENT_LIST_DIR}/hal/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/hal/*.cpp
)
message(STATUS "ACOUSTICS_SRCS_HAL: ${ACOUSTICS_SRCS_HAL}")

file(GLOB_RECURSE ACOUSTICS_SRCS_MODULE
    ${CMAKE_CURRENT_LIST_DIR}/module/*.c
    ${CMAKE_CURRENT_LIST_DIR}/module/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/module/*.cpp
)
message(STATUS "ACOUSTICS_SRCS_MODULE: ${ACOUSTICS_SRCS_MODULE}")

set(ACOUSTICS_SRCS
    ${ACOUSTICS_SRCS_API}
    ${ACOUSTICS_SRCS_CORE}
    ${ACOUSTICS_SRCS_HAL}
    ${ACOUSTICS_SRCS_MODULE}
)

get_property(ACOUSTICS_SDK_TARGET GLOBAL PROPERTY ACOUSTICS_SDK_TARGET)
message(STATUS "ACOUSTICS_SDK_TARGET: ${ACOUSTICS_SDK_TARGET}")

if(NOT DEFINED ACOUSTICS_SDK_TARGET)
    message(FATAL_ERROR "ACOUSTICS_SDK_TARGET is not defined")
endif()

if(${ACOUSTICS_SDK_TARGET} STREQUAL "POSIX")
    message(STATUS "Building for POSIX")

    add_library(acoustics STATIC ${ACOUSTICS_SRCS})

    target_include_directories(acoustics PUBLIC
        ${ACOUSTICS_INCLUDES_DIR}
    )

    target_compile_definitions(acoustics PUBLIC
        ACOUSTICS_API_VERSION=${ACOUSTICS_API_VERSION}
    )

    set_target_properties(acoustics PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
endif()

add_compile_options(-fdiagnostics-color=always -ffast-math -O2)
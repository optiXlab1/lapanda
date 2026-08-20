cmake_minimum_required(VERSION 3.16)
project(circle_acados_exact C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ACADOS_SOURCE_DIR "$ENV{ACADOS_SOURCE_DIR}" CACHE PATH "acados source/install root")
if(NOT ACADOS_SOURCE_DIR)
    message(FATAL_ERROR "Set ACADOS_SOURCE_DIR to the acados source/install root")
endif()

set(FORWARD_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code")
set(SENS_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/c_generated_sensitivity")

file(GLOB_RECURSE GENERATED_SOURCES CONFIGURE_DEPENDS
    "${FORWARD_GENERATED_DIR}/*.c"
    "${SENS_GENERATED_DIR}/*.c"
)
foreach(src ${GENERATED_SOURCES})
    get_filename_component(name "${src}" NAME)
    if(name MATCHES "^main_.*\\.c$" OR name MATCHES ".*sfunction.*\\.c$" OR name MATCHES ".*mex.*\\.c$")
        list(REMOVE_ITEM GENERATED_SOURCES "${src}")
    endif()
endforeach()

add_executable(circle_acados_exact
    main_acados.c
    ${GENERATED_SOURCES}
)

target_include_directories(circle_acados_exact PRIVATE
    "${FORWARD_GENERATED_DIR}"
    "${SENS_GENERATED_DIR}"
    "${ACADOS_SOURCE_DIR}/include"
    "${ACADOS_SOURCE_DIR}/include/acados"
    "${ACADOS_SOURCE_DIR}/include/blasfeo/include"
    "${ACADOS_SOURCE_DIR}/include/hpipm/include"
)

target_link_directories(circle_acados_exact PRIVATE "${ACADOS_SOURCE_DIR}/lib")
target_link_libraries(circle_acados_exact PRIVATE acados hpipm blasfeo qpOASES_e m)
set_target_properties(circle_acados_exact PROPERTIES
    BUILD_RPATH "${ACADOS_SOURCE_DIR}/lib"
    INSTALL_RPATH "${ACADOS_SOURCE_DIR}/lib"
)

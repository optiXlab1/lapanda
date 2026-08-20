cmake_minimum_required(VERSION 3.16)
project(rectangle_acados_gn C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ACADOS_SOURCE_DIR "$ENV{ACADOS_SOURCE_DIR}" CACHE PATH "acados source/install root")
if(NOT ACADOS_SOURCE_DIR)
    message(FATAL_ERROR "Set ACADOS_SOURCE_DIR to the acados source/install root")
endif()

set(GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code")

file(GLOB_RECURSE GENERATED_SOURCES CONFIGURE_DEPENDS "${GENERATED_DIR}/*.c")
foreach(src ${GENERATED_SOURCES})
    get_filename_component(name "${src}" NAME)
    if(name MATCHES "^main_.*\\.c$" OR name MATCHES ".*sfunction.*\\.c$" OR name MATCHES ".*mex.*\\.c$")
        list(REMOVE_ITEM GENERATED_SOURCES "${src}")
    endif()
endforeach()

add_executable(rectangle_acados_gn
    main_acados.c
    ${GENERATED_SOURCES}
)

target_include_directories(rectangle_acados_gn PRIVATE
    "${GENERATED_DIR}"
    "${ACADOS_SOURCE_DIR}/include"
    "${ACADOS_SOURCE_DIR}/include/acados"
    "${ACADOS_SOURCE_DIR}/include/blasfeo/include"
    "${ACADOS_SOURCE_DIR}/include/hpipm/include"
)

target_link_directories(rectangle_acados_gn PRIVATE "${ACADOS_SOURCE_DIR}/lib")
target_link_libraries(rectangle_acados_gn PRIVATE acados hpipm blasfeo qpOASES_e m)
set_target_properties(rectangle_acados_gn PROPERTIES
    BUILD_RPATH "${ACADOS_SOURCE_DIR}/lib"
    INSTALL_RPATH "${ACADOS_SOURCE_DIR}/lib"
)

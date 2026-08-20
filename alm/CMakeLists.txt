add_library(ALM_lib STATIC
    alm.c
)

target_link_libraries(ALM_lib PUBLIC PANDA_lib)
target_include_directories(ALM_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/alm
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/panda
)

set_property(TARGET ALM_lib PROPERTY C_STANDARD 90)
set_property(TARGET ALM_lib PROPERTY POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_lapanda src/bindings.cpp)

target_link_libraries(_lapanda PRIVATE ALM_lib PANDA_lib ${CMAKE_DL_LIBS})
if(MINGW)
    target_link_options(_lapanda PRIVATE -static-libgcc -static-libstdc++)
endif()
target_include_directories(_lapanda PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/alm
    ${PROJECT_SOURCE_DIR}/panda
)

set_target_properties(_lapanda PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lapanda
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lapanda
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lapanda
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lapanda
)

install(TARGETS _lapanda
    LIBRARY DESTINATION lapanda
    RUNTIME DESTINATION lapanda
)

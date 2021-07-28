set(COMPONENT_SRCS
    app_main.cpp
    app_model.c
    app_camera.cpp
    model.c
    printUtils.c
    )

set(COMPONENT_ADD_INCLUDEDIRS
    include
    )

set(COMPONENT_REQUIRES
    esp32-camera
    esp-face
    )

register_component()

target_link_libraries(${COMPONENT_TARGET} "-L ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib")
target_link_libraries(${COMPONENT_TARGET}
    dl
    )
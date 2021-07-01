set(COMPONENT_SRCS
    app_main.cpp
    facenet_full_prec.c
    facenet_full_prec_qu.c
    fd_forward.c
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
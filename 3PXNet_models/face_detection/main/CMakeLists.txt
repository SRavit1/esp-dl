set(COMPONENT_SRCS
    app_main.cpp
    app_facenet.c
    app_camera.cpp
    facenet_full_prec.c
    fd_forward.c
    ../3pxnet-inference-library/3pxnet/3pxnet_cn.c
    ../3pxnet-inference-library/3pxnet/3pxnet_fc.c
    ../3pxnet-inference-library/bwn/bwn_dense_cn.c
    ../3pxnet-inference-library/xnor/xnor_cn.c
    ../3pxnet-inference-library/xnor/xnor_fc.c
    ../3pxnet-inference-library/util/xnor_base.c
)

set(COMPONENT_ADD_INCLUDEDIRS
    include
    ../../../lib/include
    ../3pxnet-model-autogen
    ../3pxnet-inference-library/3pxnet
    ../3pxnet-inference-library/bwn
    ../3pxnet-inference-library/util
    ../3pxnet-inference-library/xnor
)

set(COMPONENT_REQUIRES
    esp32-camera
    esp-face
)

register_component()
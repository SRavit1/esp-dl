set(COMPONENT_SRCS
    app_main.c
    ../3pxnet-inference-library/3pxnet/3pxnet_cn.c
    ../3pxnet-inference-library/3pxnet/3pxnet_fc.c
    ../3pxnet-inference-library/bwn/bwn_dense_cn.c
    ../3pxnet-inference-library/xnor/xnor_cn.c
    ../3pxnet-inference-library/xnor/xnor_fc.c
    ../3pxnet-inference-library/util/xnor_base.c
    )

set(COMPONENT_ADD_INCLUDEDIRS
    ../../../lib/include
    ../3pxnet-model-autogen
    ../3pxnet-inference-library/3pxnet
    ../3pxnet-inference-library/bwn
    ../3pxnet-inference-library/util
    ../3pxnet-inference-library/val
    ../3pxnet-inference-library/xnor
    )

#set(COMPONENT_ADD_LDFRAGMENTS "/home/ravit/Documents/NanoCAD/esp-eye/esp-eye-development/Projects/esp-dl/3PXNet_models/face_verification/main/linker_fragment_file.lf")

register_component()

target_link_libraries(${COMPONENT_TARGET} "-L ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib")
#target_link_libraries(${COMPONENT_TARGET} dl)

set(COMPONENT_SRCS
    app_main.c
    ../3pxnet-inference-library/3pxnet/3pxnet_cn.c
    ../3pxnet-inference-library/3pxnet/3pxnet_fc.c
    ../3pxnet-inference-library/bwn/bwn_dense_cn.c
    ../3pxnet-inference-library/xnor/xnor_cn.c
    ../3pxnet-inference-library/xnor/xnor_fc.c
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

register_component()

target_link_libraries(${COMPONENT_TARGET} "-L ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib")
target_link_libraries(${COMPONENT_TARGET} dl)
#target_link_libraries(${COMPONENT_TARGET} "-L ${CMAKE_CURRENT_SOURCE_DIR}/../3pxnet-inference-library")
#target_link_libraries(${COMPONENT_TARGET} 3pxnet-inference-library)

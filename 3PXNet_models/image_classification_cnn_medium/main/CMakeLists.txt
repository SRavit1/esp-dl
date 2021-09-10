set(COMPONENT_SRCS
    app_main.c
    #../3pxnet-inference-library/3pxnet/3pxnet_cn
    #../3pxnet-inference-library/3pxnet/3pxnet_fc
    #../3pxnet-inference-library/bwn/bwn_dense_cn.c
    #../3pxnet-inference-library/util/xnor_base.c
    #../3pxnet-inference-library/xnor/xnor_cn.c
    #../3pxnet-inference-library/xnor/xnor_fc

    )

set(COMPONENT_ADD_INCLUDEDIRS
    ../include
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
target_link_libraries(${COMPONENT_TARGET} dl )
target_link_libraries(${COMPONENT_TARGET} "-L ${CMAKE_CURRENT_SOURCE_DIR}/../3pxnet-inference-library")
target_link_libraries(${COMPONENT_TARGET} 3pxnet-inference-library )
target_link_libraries(${COMPONENT_TARGET} "-L ~/.espressif/tools/xtensa-esp32-elf/esp-2019r2-8.2.0/xtensa-esp32-elf/lib/gcc/xtensa-esp32-elf/8.2.0/")
target_link_libraries(${COMPONENT_TARGET} gcc )
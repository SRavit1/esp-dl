idf_build_get_property(target IDF_TARGET)

set(lib     libcat_face_detector.a
            libhuman_face_detector.a
            libdl.a)

if(${IDF_TARGET} STREQUAL "esp32")
    set(links   "-L ${CMAKE_CURRENT_SOURCE_DIR}/lib/esp32")
elseif(${IDF_TARGET} STREQUAL "esp32s2")
    set(links   "-L ${CMAKE_CURRENT_SOURCE_DIR}/lib/esp32s2")
elseif(${IDF_TARGET} STREQUAL "esp32s3")
    set(links   "-L ${CMAKE_CURRENT_SOURCE_DIR}/lib/esp32s3")
elseif(${IDF_TARGET} STREQUAL "esp32c3")
    set(links   "-L ${CMAKE_CURRENT_SOURCE_DIR}/lib/esp32c3")
endif()

set(COMPONENT_ADD_INCLUDEDIRS   include/dl
                                include/dl/nn
                                include/dl/layer
                                include/dl/advance
                                include/model)

register_component()

target_link_libraries(${COMPONENT_TARGET}   INTERFACE   ${links})
target_link_libraries(${COMPONENT_TARGET}   INTERFACE   ${lib})


# llama-quantize
set(TARGET llama-quantize)
add_executable(${TARGET} quantize.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ../../common)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# llama-analyze-quant
if(LLAMA_BUILD_ANALYZE_QUANT)
    set(TARGET_ANALYZE llama-analyze-quant)
    add_executable(${TARGET_ANALYZE} analyze-quant.cpp)
    target_link_libraries(${TARGET_ANALYZE} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
    target_include_directories(${TARGET_ANALYZE} PRIVATE ../../common)
    target_compile_features(${TARGET_ANALYZE} PRIVATE cxx_std_17)

    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET_ANALYZE} RUNTIME)
    endif()
endif()

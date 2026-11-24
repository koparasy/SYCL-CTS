if(SYCL_CTS_ENABLE_KHR_KERNEL_PROPERTIES_TESTS)
    file(GLOB test_cases_list *.cpp)

    add_cts_test(${test_cases_list})
endif()

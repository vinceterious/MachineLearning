add_library(matrix STATIC Matrix.cpp)

################################
# Unit Tests
################################
# Add test cpp file
add_executable( runUnitTests ut/TestMatrix.cpp )
# Link test executable against gtest & gtest_main
target_link_libraries(runUnitTests gtest gtest_main)
add_test( runUnitTests runUnitTests )
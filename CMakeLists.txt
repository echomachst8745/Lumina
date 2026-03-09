cmake_minimum_required(VERSION 3.16)
project(Lumina VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LUMINA_BUILD_TESTS      "Build unit tests"   ON)
option(LUMINA_BUILD_DOCS       "Build Doxygen docs" OFF)

# ---- Static library ----
add_library(lumina STATIC
    src/ode/euler.cpp
    src/ode/runge_kutta.cpp
)

target_include_directories(lumina PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(lumina PUBLIC cxx_std_23)

# ---- Docs ----
if(LUMINA_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    doxygen_add_docs(lumina_docs
        ${PROJECT_SOURCE_DIR}/include
        ALL
        CONFIG_FILE ${PROJECT_SOURCE_DIR}/Doxyfile
        COMMENT "Generating API documentation"
    )
endif()

# ---- Tests ----
include(FetchContent)

# Redirect the FetchContent download cache to .deps/ at the project root instead of
# the default build/_deps/. This keeps the build tree free of nested .git repositories
# (GoogleTest is cloned with a full .git dir) so that `git clean -fdx` works cleanly.
# The .deps/ directory persists across clean builds, avoiding repeated downloads.
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/.deps CACHE PATH "" FORCE)

if(LUMINA_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(lumina_tests
        tests/test_vector.cpp
        tests/test_matrix.cpp
        tests/test_ode.cpp
        tests/test_solvers.cpp
        tests/test_cholesky.cpp
        tests/test_eigen.cpp
    )
    target_link_libraries(lumina_tests PRIVATE lumina GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(lumina_tests)
endif()

# ---- Install ----
install(TARGETS lumina
    ARCHIVE DESTINATION lib       # .lib / .a  (static)
    LIBRARY DESTINATION lib       # .so / .dylib (shared)
    RUNTIME DESTINATION bin       # .dll (Windows shared)
)

# Install all headers and .tpp template implementations under include/
install(DIRECTORY include/
    DESTINATION include
)

# Install Doxygen HTML docs to doc/html/ (only when built with LUMINA_BUILD_DOCS=ON)
if(LUMINA_BUILD_DOCS)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/html/
        DESTINATION doc/html
        OPTIONAL    # skip silently if docs were not generated
    )
endif()

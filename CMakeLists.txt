cmake_minimum_required(VERSION 3.20)
project(MathProject CXX)

set(CMAKE_CXX_STANDARD 17)


set(CMAKE_MACOSX_RPATH TRUE)


set(EIGEN_INCLUDE_DIR "/Users/kaushalkulkarni/Documents/Libraries/eigen-3.4.0")
include_directories(${EIGEN_INCLUDE_DIR})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")


set(PYTHON_EXECUTABLE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python3.9")
set(PYTHON_FRAMEWORK_LIB_DIR "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib")
set(PYTHON_LIBRARY "${PYTHON_FRAMEWORK_LIB_DIR}/libpython3.9.dylib")
set(PYTHON_INCLUDE_DIR "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/Headers")
set(PYTHON_FRAMEWORK_VERSION_ROOT_DIR "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9")
set(PYTHON_FRAMEWORK_PARENT_DIR "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/")

include_directories(${PYTHON_INCLUDE_DIR})

message(STATUS "Using Python Interpreter (from venv): ${PYTHON_EXECUTABLE_PATH}")
message(STATUS "Using Python Headers (from Xcode): ${PYTHON_INCLUDE_DIR}")
message(STATUS "Using Python Library (from Xcode): ${PYTHON_LIBRARY}")

# --- Main Application Target ---
# Gather all .cpp source files from subdirectories
file(GLOB_RECURSE MAIN_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Config/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Data/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Filters/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/IO/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Models/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Utils/*.cpp"
)

# Add the executable target
add_executable(MathProject ${MAIN_SOURCES})

# Link Python library to the executable
target_link_libraries(MathProject PRIVATE ${PYTHON_LIBRARY})

# --- RPATH for macOS (ensures dynamic libraries are found at runtime) ---
if (APPLE)
    # Add the venv's lib directory to RPATH so Python finds its modules
    get_filename_component(VENV_LIB_DIR "${PYTHON_EXECUTABLE_PATH}" DIRECTORY)
    get_filename_component(VENV_PARENT_DIR "${VENV_LIB_DIR}" DIRECTORY)
    set(VENV_SITE_PACKAGES_PATH "${VENV_PARENT_DIR}/lib/python3.9/site-packages") # Adjust python version if needed

    target_link_libraries(MathProject PRIVATE "-Wl,-rpath,${PYTHON_FRAMEWORK_LIB_DIR}")
    target_link_libraries(MathProject PRIVATE "-Wl,-rpath,${PYTHON_FRAMEWORK_VERSION_ROOT_DIR}")
    target_link_libraries(MathProject PRIVATE "-Wl,-rpath,${PYTHON_FRAMEWORK_PARENT_DIR}")
    target_link_libraries(MathProject PRIVATE "-Wl,-rpath,${VENV_SITE_PACKAGES_PATH}")
endif()

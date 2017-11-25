cmake_minimum_required(VERSION 2.6)
project(mtcnn_shell)

add_subdirectory(src lib)

find_package(Caffe REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDES_DIRS})
LINK_DIRECTORIES(lib)

#install(TARGETS mtcnn_shell RUNTIME DESTINATION bin)
add_executable(MTCNN_SHELL main.cpp)

add_definitions(${Caffe_DEFINITIONS})


target_link_libraries(MTCNN_SHELL ${Caffe_LIBRARIES} ${OpenCV_LIBS} MTCNN_FACE_DECT)


install(TARGETS MTCNN_SHELL RUNTIME DESTINATION install)
install(DIRECTORY image DESTINATION install)
install(DIRECTORY model DESTINATION install)

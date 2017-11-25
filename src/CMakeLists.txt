cmake_minimum_required(VERSION 2.8)  


find_package(Caffe REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDES_DIRS})



add_definitions(${Caffe_DEFINITIONS})

add_library(MTCNN_FACE_DECT SHARED detector.cpp)


target_link_libraries(MTCNN_FACE_DECT ${Caffe_LIBRARIES} ${OpenCV_LIBS})


install(TARGETS MTCNN_FACE_DECT LIBRARY DESTINATION install/lib)

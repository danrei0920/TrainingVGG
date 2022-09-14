cmake_minimum_required(VERSION 3.10)

project(TrainingVGG VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


######################QT5###########################
#QT5 INCLUDE#
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU|Clang")
set(QT_DIR "/home/danrei/Qt5.12.9/5.12.9/gcc_64")
else ()
set(QT_DIR "C:/Qt/5.15.2/msvc2019_64")
endif ()
SET(CMAKE_PREFIX_PATH ${QT_DIR})
find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

MESSAGE(STATUS "include path: ${Qt5_INCLUDE_DIRS}")
MESSAGE(STATUS "****************************************************")

######################OPEN CV#######################
#OPENCV INCLUDE#
IF (CMAKE_CXX_COMPILER_ID STREQUAL "GNU|Clang")
ELSE ()
SET (CMAKE_PREFIX_PATH  "C:/LIBS/opencv/build")
#SET (OpenCV_FOUND 1)
ENDIF ()
#link_directories(&{OpenCV_DIR}/x64/vc15/bin)
#set(OpenCV_DIR "C:/LIBS/opencv/build/x64/vc15/lib")

FIND_PACKAGE (OpenCV REQUIRED)
INCLUDE_DIRECTORIES (
        ${OpenCV_INCLUDE_DIRS}
)

#PRINT MESSAGE
MESSAGE(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "****************************************************")

######################Torch#######################
SET(CMAKE_PREFIX_PATH "C:/LIBS/libtorch-win-shared-with-deps-1.11.0+cpu/libtorch") 
find_package(Torch REQUIRED)
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#PRINT MESSAGE
MESSAGE(STATUS "include path: ${Torch_DIR}")
MESSAGE(STATUS "****************************************************")




IF (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    ENABLE_LANGUAGE("RC")
    SET (WIN32_RESOURCES logo.rc)
ENDIF()



############-------MIAN FILE-------###############
SET(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        model/vgg.cpp
        model/resnet.cpp
        model/modelcom.cpp
        dataSet.cpp
        Classification.cpp
        util/util.cpp
)



SET(
    PROJECT_HEADER
        mainwindow.h
        model/vgg.h
        model/resnet.h
        model/modelcom.h
        dataSet.h
        Classification.h
        util/util.h
)

SET(
    PROJECT_RESOURCE
        mainwindow.ui
)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_executable(${PROJECT_NAME} 
	${PROJECT_SOURCES} 
	${PROJECT_HEADER} 
	${PROJECT_RESOURCE}
    ${WIN32_RESOURCES}
 
 )

target_link_libraries(${PROJECT_NAME}  Qt5::Widgets  Qt5::Core  Qt5::Gui)
target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME}  ${TORCH_LIBRARIES})



cmake_minimum_required(VERSION 2.8)

# ------------ COMMON  ------------

FIND_PACKAGE(Boost COMPONENTS thread filesystem system REQUIRED QUIET)
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(GLUT REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

set(common_libraries  ${OpenCV_LIBRARIES})


# ------------ COMPILER FLAGS ------------

# Also add some useful compiler flag
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -O3 -std=c++11 -g -Wno-error=delete-non-virtual-dtor -Wno-error=deprecated-declarations -Wno-error=ignored-qualifiers -Wno-error=unused-parameter")

include_directories(./scenelib2/)

# ------------ SOURCES ------------

SET(SOURCES ./scenelib2/monoslam.h 
            ./scenelib2/monoslam.cpp
            ./scenelib2/motion_model.h 
            ./scenelib2/motion_model.cpp
            ./scenelib2/feature.h
            ./scenelib2/feature.cpp
            ./scenelib2/feature_init_info.h
            ./scenelib2/feature_init_info.cpp
            ./scenelib2/feature_model.h
            ./scenelib2/feature_model.cpp
            ./scenelib2/part_feature_model.h
            ./scenelib2/part_feature_model.cpp
            ./scenelib2/full_feature_model.h
            ./scenelib2/full_feature_model.cpp
            ./scenelib2/camera.h
            ./scenelib2/camera.cpp
            ./scenelib2/kalman.h
            ./scenelib2/kalman.cpp
            ./scenelib2/support/math_util.h
            ./scenelib2/support/math_util.cpp
            ./scenelib2/support/eigen_util.h
            ./scenelib2/support/eigen_util.cpp
            ./scenelib2/improc/improc.h
            ./scenelib2/improc/improc.cpp
            ./scenelib2/improc/search_multiple_overlapping_ellipses.h
            ./scenelib2/improc/search_multiple_overlapping_ellipses.cpp)
	  SET(SOURCES_GUI
            ./scenelib2/graphic/graphictool.h
            ./scenelib2/graphic/graphictool.cpp
            ./scenelib2/support/pangolin_util.h
            ./scenelib2/support/pangolin_util.cpp)
            
	  SET(SOURCES_GRABBER
            ./scenelib2/framegrabber/framegrabber.h
            ./scenelib2/framegrabber/framegrabber.cpp
            ./scenelib2/framegrabber/filegrabber.h
            ./scenelib2/framegrabber/filegrabber.cpp
            ./scenelib2/framegrabber/usbcamgrabber.h
            ./scenelib2/framegrabber/usbcamgrabber.cpp)
            
	  
	  # ----------------- ORIGINAL VERSION -----------------
	  
add_library(${appname}-original  ${SOURCES}  ./monoslam_library.cpp)
target_link_libraries(${appname}-original   ${common_libraries})	
SET_TARGET_PROPERTIES(${appname}-original PROPERTIES COMPILE_FLAGS "")
 install(TARGETS ${appname}-original  DESTINATION  lib/)





add_slambench_binary(${appname}   benchmark   original "${SOURCES_GUI};${SOURCES_GRABBER}" "" "${Pangolin_LIBRARIES};${GLUT_LIBRARIES};${OPENGL_LIBRARIES};GLU;GL;glut;${Boost_LIBRARIES};boost_program_options;boost_regex")
add_slambench_binary(${appname}   gui         original "${SOURCES_GUI};${SOURCES_GRABBER}" "" "${Pangolin_LIBRARIES};${GLUT_LIBRARIES};${OPENGL_LIBRARIES};GLU;GL;glut;${Boost_LIBRARIES};boost_program_options;boost_regex")


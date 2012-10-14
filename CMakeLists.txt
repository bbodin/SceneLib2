PROJECT("SceneLib2")

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# Overide with cmake -DCMAKE_BUILD_TYPE=Release {dir}
IF(NOT CMAKE_BUILD_TYPE)
    MESSAGE("Build type not set (defaults to debug)")
    MESSAGE("-DCMAKE_BUILD_TYPE=Release for release")
    SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

STRING(TOLOWER ${PROJECT_NAME} LIBRARY_NAME)

ADD_SUBDIRECTORY(${LIBRARY_NAME})
ADD_SUBDIRECTORY(examples)
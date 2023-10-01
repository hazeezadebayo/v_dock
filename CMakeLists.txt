cmake_minimum_required(VERSION 2.8.3)
project(v_dock) 

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED 
  COMPONENTS 
  actionlib
  actionlib_msgs
  geometry_msgs
  nav_msgs
  message_generation 
  roslib 
  roscpp
  rospy 
  std_msgs
  std_srvs
  sensor_msgs
  tf
  tf2
  rosgraph 
  rosgraph_msgs 
  move_base_msgs
  rostopic 
)

## System dependencies are found with CMake's conventions
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED) 

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

#######################################
## Declare ROS messages and services ##
#######################################

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Postgresqldb.srv
#   # Service2.srv
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  actionlib_msgs
  move_base_msgs
  std_msgs  # Or other packages containing msgs
  geometry_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME} 
  CATKIN_DEPENDS 
    actionlib
    actionlib_msgs
    geometry_msgs
    message_runtime
    nav_msgs
    roslib
    roscpp
    rospy
    sensor_msgs
    std_msgs
    tf
    move_base_msgs
    tf2
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include 
  ${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

## Declare a cpp library
# add_library(db_log
#   src/${PROJECT_NAME}/db_log.cpp
# )

# link_libraries
## Declare a cpp executable
# add_executable(DatabaseManager_ros src/DatabaseManager_ros.cpp src/DatabaseManager.cpp)
# target_link_libraries(DatabaseManager_ros
#   ${catkin_LIBRARIES} 
#   yaml-cpp
# )
# add_dependencies(DatabaseManager_ros
#   ${catkin_EXPORTED_TARGETS}
#   ${v_dock_EXPORTED_TARGETS}
# )


## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(db_log_node db_log_generate_messages_cpp)

## Specify libraries to link a library or executable target against
# target_link_libraries(db_log_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
install(DIRECTORY scripts/     # scripts/postgresqldb_client.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        USE_SOURCE_PERMISSIONS)

## Mark executables and/or libraries for installation
# install(TARGETS DatabaseManager_ros # DatabaseManager if there was morein src space and add them
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )
install(DIRECTORY config launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_db_log.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

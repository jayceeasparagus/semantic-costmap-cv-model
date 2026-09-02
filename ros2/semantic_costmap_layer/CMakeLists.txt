cmake_minimum_required(VERSION 3.8)
project(semantic_costmap_layer)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_library(semantic_costmap_layer SHARED
  src/semantic_costmap_layer.cpp
)
target_compile_features(semantic_costmap_layer PUBLIC cxx_std_17)
target_include_directories(semantic_costmap_layer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(semantic_costmap_layer
  nav2_costmap_2d
  nav_msgs
  pluginlib
  rclcpp
  tf2
  tf2_geometry_msgs
)

pluginlib_export_plugin_description_file(
  nav2_costmap_2d semantic_layer_plugin.xml
)

install(
  TARGETS semantic_costmap_layer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(FILES semantic_layer_plugin.xml DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_cost_conversion test/test_cost_conversion.cpp)
  target_link_libraries(test_cost_conversion semantic_costmap_layer)
  ament_target_dependencies(test_cost_conversion nav2_costmap_2d)
endif()

ament_export_include_directories(include)
ament_export_libraries(semantic_costmap_layer)
ament_export_dependencies(nav2_costmap_2d nav_msgs pluginlib rclcpp tf2 tf2_geometry_msgs)
ament_package()

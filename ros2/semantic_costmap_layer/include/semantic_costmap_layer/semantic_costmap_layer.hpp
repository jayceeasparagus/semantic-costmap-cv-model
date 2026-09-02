#ifndef SEMANTIC_COSTMAP_LAYER__SEMANTIC_COSTMAP_LAYER_HPP_
#define SEMANTIC_COSTMAP_LAYER__SEMANTIC_COSTMAP_LAYER_HPP_

#include <memory>
#include <mutex>
#include <string>

#include "nav2_costmap_2d/layer.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"

namespace semantic_costmap_layer
{

class SemanticCostmapLayer : public nav2_costmap_2d::Layer
{
public:
  SemanticCostmapLayer() = default;
  ~SemanticCostmapLayer() override = default;

  void onInitialize() override;
  void reset() override;
  bool isClearable() override;
  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;

  static unsigned char occupancyToCost(int8_t occupancy);
  static unsigned char mergeCost(unsigned char master, unsigned char semantic);

private:
  void gridCallback(nav_msgs::msg::OccupancyGrid::SharedPtr message);
  bool getTransform(
    const nav_msgs::msg::OccupancyGrid & grid,
    geometry_msgs::msg::TransformStamped & transform) const;

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr subscription_;
  nav_msgs::msg::OccupancyGrid::SharedPtr latest_grid_;
  mutable std::mutex grid_mutex_;
  std::string topic_;
  std::string global_frame_;
  double maximum_age_{1.0};
};

}  // namespace semantic_costmap_layer

#endif  // SEMANTIC_COSTMAP_LAYER__SEMANTIC_COSTMAP_LAYER_HPP_

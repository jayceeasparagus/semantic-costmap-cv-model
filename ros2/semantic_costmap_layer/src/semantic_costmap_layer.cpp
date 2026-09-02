#include "semantic_costmap_layer/semantic_costmap_layer.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <utility>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/time.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace semantic_costmap_layer
{

void SemanticCostmapLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("semantic layer could not lock its lifecycle node");
  }

  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("topic", rclcpp::ParameterValue("semantic_costmap"));
  declareParameter("maximum_age", rclcpp::ParameterValue(1.0));
  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".topic", topic_);
  node->get_parameter(name_ + ".maximum_age", maximum_age_);

  global_frame_ = layered_costmap_->getGlobalFrameID();
  rclcpp::SubscriptionOptions options;
  options.callback_group = callback_group_;
  subscription_ = node->create_subscription<nav_msgs::msg::OccupancyGrid>(
    topic_,
    rclcpp::QoS(1).reliable(),
    std::bind(&SemanticCostmapLayer::gridCallback, this, std::placeholders::_1),
    options);
  current_ = false;
  RCLCPP_INFO(logger_, "Semantic layer listening on %s", topic_.c_str());
}

void SemanticCostmapLayer::gridCallback(
  nav_msgs::msg::OccupancyGrid::SharedPtr message)
{
  std::lock_guard<std::mutex> lock(grid_mutex_);
  latest_grid_ = std::move(message);
  current_ = true;
}

void SemanticCostmapLayer::reset()
{
  std::lock_guard<std::mutex> lock(grid_mutex_);
  latest_grid_.reset();
  current_ = false;
}

bool SemanticCostmapLayer::isClearable()
{
  return false;
}

bool SemanticCostmapLayer::getTransform(
  const nav_msgs::msg::OccupancyGrid & grid,
  geometry_msgs::msg::TransformStamped & transform) const
{
  if (grid.header.frame_id.empty()) {
    RCLCPP_WARN(logger_, "Semantic grid has an empty frame ID");
    return false;
  }
  try {
    transform = tf_->lookupTransform(
      global_frame_, grid.header.frame_id, grid.header.stamp,
      tf2::durationFromSec(0.1));
    return true;
  } catch (const tf2::TransformException & error) {
    RCLCPP_WARN_THROTTLE(
      logger_, *clock_, 2000, "Semantic grid transform unavailable: %s", error.what());
    return false;
  }
}

void SemanticCostmapLayer::updateBounds(
  double, double, double,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) {
    return;
  }
  nav_msgs::msg::OccupancyGrid::SharedPtr grid;
  {
    std::lock_guard<std::mutex> lock(grid_mutex_);
    grid = latest_grid_;
  }
  if (!grid) {
    current_ = false;
    return;
  }
  if ((clock_->now() - rclcpp::Time(grid->header.stamp)).seconds() > maximum_age_) {
    current_ = false;
    return;
  }

  geometry_msgs::msg::TransformStamped transform;
  if (!getTransform(*grid, transform)) {
    current_ = false;
    return;
  }
  const double width = grid->info.width * grid->info.resolution;
  const double height = grid->info.height * grid->info.resolution;
  const double origin_x = grid->info.origin.position.x;
  const double origin_y = grid->info.origin.position.y;
  for (const auto & corner : {
      std::pair<double, double>{origin_x, origin_y},
      std::pair<double, double>{origin_x + width, origin_y},
      std::pair<double, double>{origin_x, origin_y + height},
      std::pair<double, double>{origin_x + width, origin_y + height}})
  {
    geometry_msgs::msg::PointStamped source;
    geometry_msgs::msg::PointStamped target;
    source.header = grid->header;
    source.point.x = corner.first;
    source.point.y = corner.second;
    tf2::doTransform(source, target, transform);
    *min_x = std::min(*min_x, target.point.x);
    *min_y = std::min(*min_y, target.point.y);
    *max_x = std::max(*max_x, target.point.x);
    *max_y = std::max(*max_y, target.point.y);
  }
  current_ = true;
}

unsigned char SemanticCostmapLayer::occupancyToCost(int8_t occupancy)
{
  if (occupancy < 0) {
    return nav2_costmap_2d::NO_INFORMATION;
  }
  const int bounded = std::clamp(static_cast<int>(occupancy), 0, 100);
  return static_cast<unsigned char>(std::lround(bounded * 254.0 / 100.0));
}

unsigned char SemanticCostmapLayer::mergeCost(
  unsigned char master, unsigned char semantic)
{
  if (semantic == nav2_costmap_2d::NO_INFORMATION) {
    return master;
  }
  if (master == nav2_costmap_2d::NO_INFORMATION) {
    return semantic;
  }
  return std::max(master, semantic);
}

void SemanticCostmapLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) {
    return;
  }
  nav_msgs::msg::OccupancyGrid::SharedPtr grid;
  {
    std::lock_guard<std::mutex> lock(grid_mutex_);
    grid = latest_grid_;
  }
  if (!grid || !current_) {
    return;
  }

  geometry_msgs::msg::TransformStamped transform;
  if (!getTransform(*grid, transform)) {
    current_ = false;
    return;
  }
  for (unsigned int row = 0; row < grid->info.height; ++row) {
    for (unsigned int column = 0; column < grid->info.width; ++column) {
      const auto index = row * grid->info.width + column;
      if (index >= grid->data.size() || grid->data[index] < 0) {
        continue;
      }
      geometry_msgs::msg::PointStamped source;
      geometry_msgs::msg::PointStamped target;
      source.header = grid->header;
      source.point.x = grid->info.origin.position.x +
        (column + 0.5) * grid->info.resolution;
      source.point.y = grid->info.origin.position.y +
        (row + 0.5) * grid->info.resolution;
      tf2::doTransform(source, target, transform);

      unsigned int map_x;
      unsigned int map_y;
      if (!master_grid.worldToMap(target.point.x, target.point.y, map_x, map_y)) {
        continue;
      }
      if (map_x < static_cast<unsigned int>(min_i) ||
        map_x >= static_cast<unsigned int>(max_i) ||
        map_y < static_cast<unsigned int>(min_j) ||
        map_y >= static_cast<unsigned int>(max_j))
      {
        continue;
      }

      const unsigned char semantic_cost = occupancyToCost(grid->data[index]);
      const unsigned char old_cost = master_grid.getCost(map_x, map_y);
      master_grid.setCost(map_x, map_y, mergeCost(old_cost, semantic_cost));
    }
  }
}

}  // namespace semantic_costmap_layer

PLUGINLIB_EXPORT_CLASS(
  semantic_costmap_layer::SemanticCostmapLayer,
  nav2_costmap_2d::Layer)

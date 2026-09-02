#include <gtest/gtest.h>

#include "nav2_costmap_2d/cost_values.hpp"
#include "semantic_costmap_layer/semantic_costmap_layer.hpp"

TEST(SemanticCostmapLayer, ConvertsOccupancyGridValues)
{
  using semantic_costmap_layer::SemanticCostmapLayer;
  EXPECT_EQ(
    SemanticCostmapLayer::occupancyToCost(-1),
    nav2_costmap_2d::NO_INFORMATION);
  EXPECT_EQ(SemanticCostmapLayer::occupancyToCost(0), 0);
  EXPECT_EQ(SemanticCostmapLayer::occupancyToCost(50), 127);
  EXPECT_EQ(SemanticCostmapLayer::occupancyToCost(100), 254);
}

TEST(SemanticCostmapLayer, NeverLowersExistingMasterCost)
{
  using semantic_costmap_layer::SemanticCostmapLayer;
  EXPECT_EQ(SemanticCostmapLayer::mergeCost(254, 100), 254);
  EXPECT_EQ(SemanticCostmapLayer::mergeCost(80, 220), 220);
  EXPECT_EQ(
    SemanticCostmapLayer::mergeCost(nav2_costmap_2d::NO_INFORMATION, 220),
    220);
  EXPECT_EQ(
    SemanticCostmapLayer::mergeCost(80, nav2_costmap_2d::NO_INFORMATION),
    80);
}

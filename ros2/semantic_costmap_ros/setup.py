from setuptools import find_packages, setup


package_name = "semantic_costmap_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/semantic_costmap.launch.py"]),
        (f"share/{package_name}/config", ["config/semantic_costmap.yaml"]),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Jayce",
    maintainer_email="jayce@example.com",
    description="ROS 2 semantic costmap perception node",
    license="MIT",
    entry_points={
        "console_scripts": [
            "semantic_costmap_node = semantic_costmap_ros.semantic_costmap_node:main",
            "semantic_map_accumulator = semantic_costmap_ros.semantic_map_accumulator_node:main",
        ],
    },
)

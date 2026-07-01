import os
from glob import glob

from setuptools import setup

package_name = "lidar_nav2"

setup(
    name=package_name,
    version="0.1.0",
    # Ship the ROS node package and the (ROS-free) algorithm package together.
    packages=[package_name, "lidar_pathplan"],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="sanjitk",
    maintainer_email="akpostme@gmail.com",
    description="Lidar 2.5D discretization published as a nav2 OccupancyGrid for a quadruped.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "costmap_node = lidar_nav2.costmap_node:main",
        ],
    },
)

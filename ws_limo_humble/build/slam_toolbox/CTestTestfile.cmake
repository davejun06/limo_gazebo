# CMake generated Testfile for 
# Source directory: /home/davejun06/LIMO_GAZEBO/ws_limo_humble/src/slam_toolbox
# Build directory: /home/davejun06/LIMO_GAZEBO/ws_limo_humble/build/slam_toolbox
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(map_saver_test "/usr/bin/python3" "-u" "/opt/ros/humble/share/ament_cmake_test/cmake/run_test.py" "/home/davejun06/LIMO_GAZEBO/ws_limo_humble/build/slam_toolbox/test_results/slam_toolbox/map_saver_test.gtest.xml" "--package-name" "slam_toolbox" "--output-file" "/home/davejun06/LIMO_GAZEBO/ws_limo_humble/build/slam_toolbox/ament_cmake_gtest/map_saver_test.txt" "--command" "/home/davejun06/LIMO_GAZEBO/ws_limo_humble/build/slam_toolbox/map_saver_test" "--gtest_output=xml:/home/davejun06/LIMO_GAZEBO/ws_limo_humble/build/slam_toolbox/test_results/slam_toolbox/map_saver_test.gtest.xml")
set_tests_properties(map_saver_test PROPERTIES  LABELS "gtest" REQUIRED_FILES "/home/davejun06/LIMO_GAZEBO/ws_limo_humble/build/slam_toolbox/map_saver_test" TIMEOUT "60" WORKING_DIRECTORY "/home/davejun06/LIMO_GAZEBO/ws_limo_humble/build/slam_toolbox" _BACKTRACE_TRIPLES "/opt/ros/humble/share/ament_cmake_test/cmake/ament_add_test.cmake;125;add_test;/opt/ros/humble/share/ament_cmake_gtest/cmake/ament_add_gtest_test.cmake;86;ament_add_test;/opt/ros/humble/share/ament_cmake_gtest/cmake/ament_add_gtest.cmake;93;ament_add_gtest_test;/home/davejun06/LIMO_GAZEBO/ws_limo_humble/src/slam_toolbox/CMakeLists.txt;190;ament_add_gtest;/home/davejun06/LIMO_GAZEBO/ws_limo_humble/src/slam_toolbox/CMakeLists.txt;0;")
subdirs("lib/karto_sdk")
subdirs("slam_toolbox__py")
subdirs("slam_toolbox__rs")
subdirs("gtest")

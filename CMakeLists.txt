cmake_minimum_required(VERSION 3.29)
project(OpenCV_test)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 包含 OpenCV 的头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行目标
add_executable(OpenCV_test main.cpp
        include/Light.h
        src/Light.cpp
        src/KNN.cpp
        include/Armor.h
        src/Armor.cpp
        include/KNN.h
        include/Judge_Light.h
        src/Judge_Light.cpp)

# 链接 OpenCV 库
target_link_libraries(OpenCV_test ${OpenCV_LIBS})

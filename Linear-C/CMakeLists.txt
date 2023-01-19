cmake_minimum_required(VERSION 3.22)
project(Linear_C)

set(CMAKE_CXX_STANDARD 14)

add_executable(Linear_C main.c src/neuron.c src/layer.c src/backprop.c src/read_csv.c)

'''
This file is for Part 3:
Download, install, and learn to use the Python motion planning library at 
https://github. com/motion-planning/rrt-algorithms. Examples are available in the repository.
I experimented with rrt 3d
'''

import numpy as np
import uuid
import sys
sys.path.append('/Users/tina/Desktop/ECE276B_PR2/starter_code/rrt-algorithms-develop')

from src.rrt.rrt import RRT
from src.search_space.search_space import SearchSpace
from src.utilities.obstacle_generation import generate_random_obstacles
from src.utilities.plotting import Plot
from main import load_map

Q = np.array([(8, 4)])  # length of tree edges
r = 0.1  # length of smallest edge to check for intersection with obstacles
max_samples = 4092 * 8 # max number of samples to take before timing out
prc = 0.2  # probability of checking for a connection to goal

def add_blocks(X, blocks):
    for i in range(len(blocks)):
        min_corner = np.empty(X.dimensions, np.float)
        max_corner = np.empty(X.dimensions, np.float)
        for j in range(X.dimensions):
            min_corner[j] = blocks[i][j]
            max_corner[j] = blocks[i][j + 3]

        obstacle = np.append(min_corner, max_corner)
        X.obs.add(uuid.uuid4(), tuple(obstacle), tuple(obstacle))

def plot_path(filename, X, rrt, path, blocks, x_init, x_goal):
    plot = Plot(filename)
    plot.plot_tree(X, rrt.trees)
    if path is not None:
        plot.plot_path(X, path)
    plot.plot_obstacles(X, blocks)
    plot.plot_start(X, x_init)
    plot.plot_goal(X, x_goal)
    plot.draw(auto_open=True)

def test_single_cube():
    boundary, blocks = load_map('/Users/tina/Desktop/ECE276B_PR2/starter_code/maps/single_cube.txt')
    x_min = boundary[0][0]
    y_min = boundary[0][1]
    z_min = boundary[0][2]
    x_max = boundary[0][3]
    y_max = boundary[0][4]
    z_max = boundary[0][5]
    x_init = (2.3, 2.3, 1.3)
    x_goal = (7.0, 7.0, 5.5)
    X_dimensions = np.array([(x_min, x_max), (y_min, y_max), (z_min, z_max)])  # dimensions of Search Space

    X = SearchSpace(X_dimensions)
    add_blocks(X, blocks)
    rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
    path = rrt.rrt_search()

    # plot
    plot_path("single_cube", X, rrt, path, blocks, x_init, x_goal)


def test_maze():
    boundary, blocks = load_map('/Users/tina/Desktop/ECE276B_PR2/starter_code/maps/maze.txt')
    x_min = boundary[0][0]
    y_min = boundary[0][1]
    z_min = boundary[0][2]
    x_max = boundary[0][3]
    y_max = boundary[0][4]
    z_max = boundary[0][5]
    x_init = (0.0, 0.0, 1.0)
    x_goal = (12.0, 12.0, 5.0)
    X_dimensions = np.array([(x_min, x_max), (y_min, y_max), (z_min, z_max)])  # dimensions of Search Space

    X = SearchSpace(X_dimensions)
    add_blocks(X, blocks)
    rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
    path = rrt.rrt_search()

    # plot
    plot_path("maze", X, rrt, path, blocks, x_init, x_goal)

def test_window():
    boundary, blocks = load_map('/Users/tina/Desktop/ECE276B_PR2/starter_code/maps/window.txt')
    x_min = boundary[0][0]
    y_min = boundary[0][1]
    z_min = boundary[0][2]
    x_max = boundary[0][3]
    y_max = boundary[0][4]
    z_max = boundary[0][5]
    x_init = (0.2, -4.9, 0.2)
    x_goal = (6.0, 18.0, 3.0)
    X_dimensions = np.array([(x_min, x_max), (y_min, y_max), (z_min, z_max)])  # dimensions of Search Space

    X = SearchSpace(X_dimensions)
    add_blocks(X, blocks)
    rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
    path = rrt.rrt_search()

    # plot
    plot_path("window", X, rrt, path, blocks, x_init, x_goal)

def test_tower():
    boundary, blocks = load_map('/Users/tina/Desktop/ECE276B_PR2/starter_code/maps/tower.txt')
    x_min = boundary[0][0]
    y_min = boundary[0][1]
    z_min = boundary[0][2]
    x_max = boundary[0][3]
    y_max = boundary[0][4]
    z_max = boundary[0][5]
    x_init = (2.5, 4.0, 0.5)
    x_goal = (4.0, 2.5, 19.5)
    X_dimensions = np.array([(x_min, x_max), (y_min, y_max), (z_min, z_max)])  # dimensions of Search Space

    X = SearchSpace(X_dimensions)
    add_blocks(X, blocks)
    rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
    path = rrt.rrt_search()

    # plot
    plot_path("tower", X, rrt, path, blocks, x_init, x_goal)

def test_flappy_bird():
    boundary, blocks = load_map('/Users/tina/Desktop/ECE276B_PR2/starter_code/maps/flappy_bird.txt')
    x_min = boundary[0][0]
    y_min = boundary[0][1]
    z_min = boundary[0][2]
    x_max = boundary[0][3]
    y_max = boundary[0][4]
    z_max = boundary[0][5]
    x_init = (0.5, 2.5, 5.5)
    x_goal = (19.0, 2.5, 5.5)
    X_dimensions = np.array([(x_min, x_max), (y_min, y_max), (z_min, z_max)])  # dimensions of Search Space

    X = SearchSpace(X_dimensions)
    add_blocks(X, blocks)
    rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
    path = rrt.rrt_search()

    # plot
    plot_path("flappy_bird", X, rrt, path, blocks, x_init, x_goal)


def test_room():
    boundary, blocks = load_map('/Users/tina/Desktop/ECE276B_PR2/starter_code/maps/room.txt')
    x_min = boundary[0][0]
    y_min = boundary[0][1]
    z_min = boundary[0][2]
    x_max = boundary[0][3]
    y_max = boundary[0][4]
    z_max = boundary[0][5]
    x_init = (1.0, 5.0, 1.5)
    x_goal = (9.0, 7.0, 1.5)
    X_dimensions = np.array([(x_min, x_max), (y_min, y_max), (z_min, z_max)])  # dimensions of Search Space

    X = SearchSpace(X_dimensions)
    add_blocks(X, blocks)
    rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
    path = rrt.rrt_search()

    # plot
    plot_path("room", X, rrt, path, blocks, x_init, x_goal)

def test_monza():
    boundary, blocks = load_map('/Users/tina/Desktop/ECE276B_PR2/starter_code/maps/monza.txt')
    x_min = boundary[0][0]
    y_min = boundary[0][1]
    z_min = boundary[0][2]
    x_max = boundary[0][3]
    y_max = boundary[0][4]
    z_max = boundary[0][5]
    x_init = (0.5, 1.0, 4.9)
    x_goal = (3.8, 1.0, 0.1)
    X_dimensions = np.array([(x_min, x_max), (y_min, y_max), (z_min, z_max)])  # dimensions of Search Space

    X = SearchSpace(X_dimensions)
    add_blocks(X, blocks)
    rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
    path = rrt.rrt_search()

    # plot
    plot_path("monza", X, rrt, path, blocks, x_init, x_goal)



test_single_cube()
test_maze()
test_window()
test_tower()
test_flappy_bird()
test_room()
test_monza()


import math

import numpy as np
import matplotlib.pyplot as plt
from twoD.environment import MapEnvironment
from twoD.building_blocks import BuildingBlocks2D
from twoD.prm import PRMController
from twoD.visualizer import Visualizer
from threeD.environment import Environment
from threeD.kinematics import UR5e_PARAMS, Transform
from threeD.building_blocks import BuildingBlocks3D
from threeD.visualizer import Visualize_UR


def test_building_blocks():
    planning_env = MapEnvironment(json_file="./twoD/map_mp.json")
    bb = BuildingBlocks2D(planning_env)
    visualizer = Visualizer(bb)
    
    
    # Test compute_distance
    conf1 = np.array([0.78, -0.78, 0.0, 0.0])
    conf2 = np.array([0.8, -0.8, 0.8, 0.5])
    config3 = np.array([0.5, 0.5, 2.0, 3.0]) #collision config
    distance = bb.compute_distance(conf1, conf2)
    print(f"Distance between configs: {distance}")
    
    # Test compute_forward_kinematics
    robot_pos = bb.compute_forward_kinematics(conf1)
    print(f"Robot positions:\n{robot_pos}")
    
    # Test validate_robot with valid config
    print(f"Valid config check: {bb.validate_robot(robot_pos)}")
    
    # Test validate_robot with self-collision (create overlapping config)
    collision_conf = np.array([0.0, 0.0, 0.0, np.pi])  # Should cause self-collision
    collision_pos = bb.compute_forward_kinematics(collision_conf)
    collision_pos2 = bb.compute_forward_kinematics(config3)
    print("Testing self-collision scenarios:")
    print(f"Self-collision check: {bb.validate_robot(collision_pos)}")
    print("collision 1 end")# should be false
    print(f"Self-collision check: {bb.validate_robot(collision_pos2)}")# should be false

    # Test config_validity_checker
    print(f"Config validity: {bb.config_validity_checker(conf1)}")
    visualizer.visualize_map(config=conf1)
    visualizer.visualize_map(config=collision_conf)
    visualizer.visualize_map(config=config3)



def run_2d():
    conf = np.array([0.78, -0.78, 0.0, 0.0])

    # prepare the map
    planning_env = MapEnvironment(json_file="./twoD/map_mp.json")
    bb = BuildingBlocks2D(planning_env)
    visualizer = Visualizer(bb)

    robot_positions = bb.compute_forward_kinematics(given_config=conf)
    print(bb.validate_robot(robot_positions=robot_positions)) # check robot validity
    print(bb.config_validity_checker(config=conf)) # check robot and map validity

    visualizer.visualize_map(config=conf)


def run_prm():
    conf1 = np.array([0.78, -0.78, 0.0, 0.0])
    conf2 = np.array([0.8, -0.8, 0.8, 0.5])

    planning_env = MapEnvironment(json_file="./twoD/map_mp.json")
    bb = BuildingBlocks2D(planning_env)
    visualizer = Visualizer(bb)
    prm = PRMController(conf1, conf2, bb)

    plan = prm.run_PRM(num_coords=500, k=10) # TODO: HW2 4.3.5
    if plan is None:
        print("No path found.")
        return
    
    print(bb.compute_path_cost(plan))
    xy_path = np.array([
        bb.compute_forward_kinematics(q)[-1]
        for q in plan
    ])

    # Visualize static map
    visualizer.visualize_map(config=conf1, plan=xy_path)
    visualizer.visualize_plan_as_gif(plan)
    
def plot_P1(prm):
    results = prm.create_graph(100, 100, 7)

    plt.figure(figsize=(10,6))
    for key in results:
        ns = [item[0] for item in results[key]]
        costs = [item[2] for item in results[key]]
        plt.plot(ns, costs, marker='o', label=key)

    plt.title("P1: Path Cost vs n")
    plt.xlabel("Number of Milestones n")
    plt.ylabel("Path Cost")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_P2(prm): #Path Cost vs Runtime
    results = prm.create_graph(100, 100, 7)

    plt.figure(figsize=(10,6))
    for key in results:
        runtimes = [item[1] for item in results[key]]
        costs = [item[2] for item in results[key]]
        plt.plot(runtimes, costs, marker='o', label=key)

    plt.title("P2: Path Cost vs Runtime")
    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Path Cost")
    plt.grid(True)
    plt.legend()
    plt.show()



def generate_graph():
    conf1 = np.array([0.78, -0.78, 0.0, 0.0])
    conf2 = np.array([0.8, -0.8, 0.8, 0.5])
    planning_env = MapEnvironment(json_file="./twoD/map_mp.json")
    bb = BuildingBlocks2D(planning_env)
    prm = PRMController(conf1, conf2, bb)
    plot_P1(prm)



def run_3d():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    transform = Transform(ur_params)
    env = Environment(env_idx=1)
    bb = BuildingBlocks3D(transform=transform,
                          ur_params=ur_params,
                          resolution=0.1,
                          env=env)

    

    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)

    # --------- configurations-------------
    conf1 = np.deg2rad([0, -90, 0, -90, 0, 0])

    conf2 = np.array([-0.694, -1.376, -2.212, -1.122, 1.570, -2.26])

    # ---------------------------------------

    # collision checking examples
    res = bb.config_validity_checker(conf=conf1)
    print("Configuration 1 is free collision:", res)
    res = bb.edge_validity_checker(prev_conf=conf1 ,current_conf=conf2)
    print("Edge between conf 1 and conf 2 is free collision:", res)

    visualizer.show_conf(conf1)

if __name__ == "__main__":
    #test_building_blocks()
    #run_2d()
    #run_prm()
    # run_3d()
    generate_graph()

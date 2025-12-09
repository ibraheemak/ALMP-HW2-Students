import time

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import math
import matplotlib.pyplot as plt

class PRMController:
    def __init__(self, start, goal, bb):
        self.graph = nx.Graph()
        self.bb = bb
        self.start = start
        self.goal = goal
        self.configs = []  # list of all configurations (nodes) added to the graph


        self.coordinates_history = []
        # Feel free to add class variables as you wish

    def run_PRM(self, num_coords=100, k=5):
        """
            find a plan to get from current config to destination
            return-the found plan and None if couldn't find
        """
        path = []
        # TODO: HW2 4.3.4
        #Sample milestones
        milestones = self.gen_coords(num_coords)

        #Add start + goal to the list
        all_configs = np.vstack([milestones, self.start, self.goal])

        #Build the roadmap over ALL configurations
        self.add_to_graph(all_configs, k)

        #Find node IDs of start and goal
        start_id = len(all_configs) - 2
        goal_id = len(all_configs) - 1

        #Dijkstra shortest path
        try:
            node_path = nx.shortest_path(
                self.graph,
                source=start_id,
                target=goal_id,
                weight="weight"
            )
        except nx.NetworkXNoPath:
            return None

        #Convert node IDs to configs
        config_path = [self.configs[n] for n in node_path]
        return config_path
    
    def create_graph(self, base_number, how_many_to_add, num_searches):
        """
        Creates all 5 PRM curves (for k = 5, 10, log(n), 10log(n), n/10)
        Returns a dictionary mapping each k-type to a list of (n, runtime, cost).
        """

        results = {
            "k=5": [],
            "k=10": [],
            "k=log(n)": [],
            "k=10log(n)": [],
            "k=n/10": []
        }

        # Generate ALL samples ONCE
        total_needed = base_number + how_many_to_add * (num_searches - 1)
        all_samples = self.gen_coords(total_needed)

        for i in range(num_searches):
            n = base_number + i * how_many_to_add      # 100, 200, ..., 700
            milestones = all_samples[:n]              

            # build config array
            configs = np.vstack([milestones, self.start, self.goal])

        
            k_values = {
                "k=5": 5,
                "k=10": 10,
                "k=log(n)": max(1, int(np.log(n))),
                "k=10log(n)": max(1, int(10*np.log(n))),
                "k=n/10": max(1, int(n/10))
            }

            for key, k in k_values.items():

                # reset graph
                self.graph = nx.Graph()
                self.configs = []

                # build roadmap
                t0 = time.time()
                self.add_to_graph(configs, k)
                path = self.shortest_path()
                runtime = time.time() - t0

                if path is None:
                    cost = float('inf')
                else:
                    cost = self.bb.compute_path_cost(path)

                results[key].append((n, runtime, cost))

        return results




    def gen_coords(self, n=5):
        """
        Generate 'n' random collision-free samples called milestones.
        n: number of collision-free configurations to generate
        """
        # TODO: HW2 4.3.1
        samples = []
        dim = self.bb.dim   # number of joints (should be 4)

        while len(samples) < n:
            # sample random configuration in C-space
            q = np.random.uniform(low=-np.pi, high=np.pi, size=dim)

            # keep only if collision-free
            if self.bb.config_validity_checker(q):
                samples.append(q)

        samples = np.array(samples)
        self.coordinates_history.append(samples)  # useful for plots later
        return samples



    def add_to_graph(self, configs, k):
        """
            add new configs to the graph.
        """
        # TODO: HW2 4.3.2
        #add all new configs as graph nodes
        for q in configs:
            node_id = len(self.configs)
            self.configs.append(np.array(q))
            self.graph.add_node(node_id, config=self.configs[-1])
        
        #build KDTree on all nodes
        data = np.vstack(self.configs)
        tree = KDTree(data)

        #find k nearest neighbours for each node
        for i, q in enumerate(self.configs):
            dists, idxs = tree.query(q, k=k+1)  # includes itself
            idxs = np.atleast_1d(idxs)  # ensure array format even if k=1

            # remove itself, keep k others
            neighbours = [j for j in idxs if j != i][:k]

            # Check edge validity
            for j in neighbours:
                q2 = self.configs[j]

                if self.bb.edge_validity_checker(q, q2):
                    w = self.bb.compute_distance(q, q2)
                    self.graph.add_edge(i, j, weight=w)

    def find_nearest_neighbour(self, config, k=5):
        """
            Find the k nearest neighbours to config
            return-list of indices of the k nearest neighbours
        """
        # TODO: HW2 4.3.2
        if len(self.configs) == 0:
            return []
        # Build KDTree over existing configs
        data = np.vstack(self.configs)  # shape (num_configs, dim)
        tree = KDTree(data)

        # We ask for k+1 neighbours then remove the zero-distance one.
        num_query = min(k + 1, len(self.configs))
        dists, idxs = tree.query(config, k=num_query)

        # Ensure we always work with arrays (even if k=1)
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)

        # Filter out self (distance ≈ 0) and keep up to k others
        neighbours = [
            int(idx)
            for idx, d in zip(idxs, dists)
            if d > 1e-8   # exclude self if present
        ]

        return neighbours[:k]
    


    def shortest_path(self):
        """
            Find the shortest path from start to goal using Dijkstra's algorithm (you can use previous implementation from HW1)'
        """
        # TODO: HW2 4.3.3
        
        # find the indices of start and goal in self.configs
        start_id = None
        goal_id = None

        for i, q in enumerate(self.configs):
            if np.allclose(q, self.start):
                start_id = i
            if np.allclose(q, self.goal):
                goal_id = i

        if start_id is None or goal_id is None:
            print("Start or goal not found in graph nodes.")
            return None

        try:
            # find shortest path of node IDs using Dijkstra
            node_path = nx.shortest_path(
                self.graph,
                source=start_id,
                target=goal_id,
                weight="weight"
            )
        except nx.NetworkXNoPath:
            return None

        # convert node IDs to configuration vectors
        config_path = [self.configs[n] for n in node_path]

        return config_path


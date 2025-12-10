import numpy as np
from kinematics import Transform

class BuildingBlocks3D(object):
    '''
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    '''

    def __init__(self, transform, ur_params, env, resolution=0.1):
        self.transform = transform
        self.ur_params = ur_params
        self.env = env
        self.resolution = resolution
        # self.p_bias = p_bias
        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07, 0.05])

        self.single_mechanical_limit = list(self.ur_params.mechamical_limits.values())[-1][-1]

        # pairs of links that can collide during sampling
        self.possible_link_collisions = [['shoulder_link', 'forearm_link'],
                                         ['shoulder_link', 'wrist_1_link'],
                                         ['shoulder_link', 'wrist_2_link'],
                                         ['shoulder_link', 'wrist_3_link'],
                                         ['upper_arm_link', 'wrist_1_link'],
                                         ['upper_arm_link', 'wrist_2_link'],
                                         ['upper_arm_link', 'wrist_3_link'],
                                         ['forearm_link', 'wrist_2_link'],
                                         ['forearm_link', 'wrist_3_link']]

    def sample_random_config(self, goal_prob,  goal_conf) -> np.array:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        :param goal_prob - the probability that goal should be sampled
        """
        # TODO: HW2 5.2.1
        dim = 3
    #     generate random number between 0 and 1, and act according to goal prob or not
        rng = np.random.uniform(low=0.0, high=1.0, size=None)
        if rng <= goal_prob:
            return goal_conf
        else:
            while(1):
                q = np.random.uniform(low=-np.pi, high=np.pi, size=dim)
                free = True
                for i in range(len(q)):
                    # check if in interval. TODO: figure out the mechanical limits format (this may be incorrect)
                    if (q[i] < self.single_mechanical_limit[i][0] or
                             q[i] > self.single_mechanical_limit[i][1]):
                        # take new sample
                        free = False
                        break
                if free:
                    return q

    def config_validity_checker(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return False if in collision
        @param conf - some configuration
        """
        # TODO: HW2 5.2.2
        radius = 1 # TODO: where do we get the radius??
        cols = self.possible_link_collisions
        # go over every pair in cols and check if they collide by iterating on all spheres
        # note: this can be done more efficiently than iterate on all spheres.
        for i in range(len(cols)):
            for coord1 in conf[cols[i][0]]:
                for coord2 in conf[cols[i][1]]:
                    # check if collide
                    # TODO: make sure this is correct function for collision check.
                    if np.linalg.norm(coord1 - coord2) < 2*radius:
                        return False
        # check collisions with the floor
        links_list = [] # TODO: where to get list of all of the arms?
        for link in links_list:
            for sphere in link:
                if sphere[2] < radius: # TODO: check that this is the Z coord
                    return False
        # TODO: check collisiosn with obstacles (how? where are they stored?)
        return True

    def edge_validity_checker(self, prev_conf, current_conf) -> bool:
        '''check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        '''
        # TODO: HW2 5.2.4
        min_configs = 2
        # TODO: why is resolution float? get the right int from it somehow
        num_configs = max(min_configs, self.resolution)
        configs = []
        for i in range(num_configs):
            frac = i/(num_configs-1) # e.g. n=4 we want [0, 1/3, 2/3, 1].
            conf = prev_conf*(1-frac) + current_conf*frac # TODO: check that this is the right way to do it
            if (not self.config_validity_checker(conf)):
                return False
        return True

    def compute_distance(self, conf1, conf2):
        '''
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        '''
        return np.dot(self.cost_weights, np.power(conf1 - conf2, 2)) ** 0.5

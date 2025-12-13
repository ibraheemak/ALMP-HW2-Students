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
        self.   possible_link_collisions = [['shoulder_link', 'forearm_link'],
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
        # HW2 5.2.1
    #     generate random number between 0 and 1, and act according to goal prob or not
        uni_sample = np.random.uniform(low=0.0, high=1.0, size=None)
        if uni_sample <= goal_prob:
            return goal_conf
        else:
            # sample a random dist with the same size as the goal conf, limited by the mechanical limit.
            return np.random.uniform(low=-self.single_mechanical_limit, high=self.single_mechanical_limit, size=len(goal_conf))

    def config_validity_checker(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return False if in collision
        @param conf - some configuration
        """
        # HW2 5.2.2

        collisions = self.possible_link_collisions
        radii = self.ur_params.sphere_radius
        sphere_coords = self.ur_params.conf2sphere_coords(conf) # figure out the location of every link using transforms
        for coli in collisions:
            link1, link2 = coli
            # check that the spheres dont intersect
            for s1 in sphere_coords[link1]: # for sphere in link1
                for s2 in sphere_coords[link2]: # for sphere in link2
                    if np.linalg.norm(s1 - s2) < radii[link1] + radii[link2]:
                        return False
        # check collisions with the floor
        links = self.ur_params.ur_links
        obstacles, obs_radius = self.env.obstacles, self.env.radius
        for link in links:
            for sphere in sphere_coords[link]:
                # check collision with the floor
                if sphere[2] < radii[link]: # TODO: check that this is the Z coord
                    return False
                # check collision with obstacles
                for obs in obstacles:
                    if np.linalg.norm(sphere-obs) < radii[link] + obs_radius:
                        return False
        # did not find collisions of any kind. return true.
        return True

    def edge_validity_checker(self, prev_conf, current_conf) -> bool:
        '''check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        '''
        # HW2 5.2.4
        # TODO: find out what he meant by config resolution
        # min_configs = 2
        # num_configs = np.ceil(1/self.resolution)
        # num_configs = max(min_configs, num_configs)
        # for i in range(num_configs):
        #     frac = i/(num_configs-1) # e.g. n=4 we want [0, 1/3, 2/3, 1].
        #     conf = prev_conf*(1-frac) + current_conf*frac
        #     if not self.config_validity_checker(conf):
        #         return False
        res = min(1.0, self.resolution)
        progress = 0
        while True:
            conf = prev_conf * (1.0 - progress) + current_conf * progress  # TODO: check that this is the right way to use the resolution
            if not self.config_validity_checker(conf):
                return False
            if progress == 1.0:
                return True
            progress += res
            progress = min(progress, 1.0) # do one last iteration, for the final config.

    def compute_distance(self, conf1, conf2):
        '''
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        '''
        return np.dot(self.cost_weights, np.power(conf1 - conf2, 2)) ** 0.5

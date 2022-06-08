import numpy as np


class MonteCarlo():

    def __init__(self, model, theta_distribution): 
        self.model = model 
        self.theta_distribution = theta_distribution

    def sample(self, num_samples):
        result = np.zeros(num_samples)
        for i in range(num_samples):
            theta = np.random.choice(self.theta_distribution)
            result[i] = self.model.evaluate_distance(theta) 
        return result
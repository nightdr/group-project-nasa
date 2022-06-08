import numpy as np


class MonteCarlo():

    def __init__(self, model, theta_distribution): 
        self.model = model 
        self.theta_distribution = theta_distribution

    def sample(self, num_samples):
        result = np.zeros(num_samples)
        

        return result
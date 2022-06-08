class SurrogateModel:
    def __init__(self, method):
        self.method = method

    def evaluate_distance(self, theta):
        return float(self.method.predict(theta))

    def train_surrogate(self, hdf5_file):
        self.method.fit(hdf5_file['\/']['explicit_model']['theta'], hdf5_file['\/']['explicit_model']['distance'])


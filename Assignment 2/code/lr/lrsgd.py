# Do not use anything outside of the standard distribution of python
# when implementing this class
import math 

class LogisticRegressionSGD:
    """
    Logistic regression with stochastic gradient descent
    """

    def __init__(self, eta, mu, n_feature):
        """
        Initialization of model parameters
        """
        self.eta = eta
        self.weight = [0.0] * n_feature
        self.mu = mu

    def fit(self, X, y):
        """
        Update model using a pair of training sample
        """
        updateModel = 0

        model1 = self.weight[:]

        for dataPoints in X:
            updateModel = updateModel + self.weight[dataPoints[0]] * dataPoints[1]

        for dataPoints in X:
            updateModel2 = dataPoints[1] * (y-1 / (1 + math.exp(-updateModel)))
            model1[dataPoints[0]] = model1[dataPoints[0]] + self.eta * updateModel2

        for dataPoints in range(len(self.weight)):
            model1[dataPoints] = model1[dataPoints] - self.eta * self.mu * (self.weight[dataPoints] * 2)

        self.weight = model1


    def predict(self, X):
        """
        Predict 0 or 1 given X and the current weights in the model
        """
        return 1 if predict_prob(X) > 0.5 else 0

    def predict_prob(self, X):
        """
        Sigmoid function
        """
        return 1.0 / (1.0 + math.exp(-math.fsum((self.weight[f]*v for f, v in X))))

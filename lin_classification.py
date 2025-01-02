import numpy as np

def transform_labels(y):
    return np.where(y == 1, 1, -1).reshape(-1, 1)

def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.hstack((intercept, X))

class LinearClassifier:
    def __init__(self, loss='mse', learning_rate=0.01, n_iterations=1000, lambda1=0.0, lambda2=0.0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.w = None
        self.loss_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []

    def loss_and_gradient(self, X, y):
        m = X.shape[0]
        scores = X @ self.w
        margins = y * scores

        if self.loss == 'mse':
            loss = (1 - margins) ** 2
            gradient = -(X.T @ (y * (1 - margins))) / m
        elif self.loss == 'logistic':
            loss = np.log(1 + np.exp(-margins))
            sigmoid = 1 / (1 + np.exp(-margins))
            gradient = -(X.T @ (y * (1 - sigmoid))) / m
        elif self.loss == 'exponential':
            loss = np.exp(-margins)
            gradient = -(X.T @ (y * loss)) / m
        else:
            raise ValueError("Unsupported loss function")

        gradient += self.lambda1 * np.sign(self.w) + 2 * self.lambda2 * self.w
        loss_mean = np.mean(loss) + self.lambda1 * np.sum(np.abs(self.w)) + self.lambda2 * np.sum(self.w ** 2)
        return loss_mean, gradient

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        m, n = X_train.shape
        self.w = np.zeros((n, 1))

        for i in range(self.n_iterations):
            loss, gradient = self.loss_and_gradient(X_train, y_train)
            self.w -= self.learning_rate * gradient
            self.loss_history.append(loss)

            if X_test is not None and y_test is not None:
                test_loss, _ = self.loss_and_gradient(X_test, y_test)
                self.test_loss_history.append(test_loss)

                y_pred_test = self.predict(X_test)
                test_accuracy = self.evaluate_accuracy(y_test, y_pred_test)
                self.test_accuracy_history.append(test_accuracy)

    def predict(self, X):
        linear_output = X @ self.w
        y_pred = np.sign(linear_output)
        y_pred[y_pred == 0] = 1
        return y_pred

    def evaluate_accuracy(self, y_true, y_pred):
        return np.mean(y_true.flatten() == y_pred.flatten())

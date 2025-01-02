import numpy as np

def kernel_function(X1, X2, kernel_type='linear', degree=3, gamma=None):
    if gamma is None:
        gamma = 1 / X1.shape[1]

    if kernel_type == 'linear':
        return X1 @ X2.T
    elif kernel_type == 'polynomial':
        return (X1 @ X2.T + 1) ** degree
    elif kernel_type == 'rbf':
        sq_dists = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
                   np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-gamma * sq_dists)
    else:
        raise ValueError("Неподдерживаемый тип ядра")


class SVM_SMO:
    def __init__(self, kernel='linear', C=0.1, tol=1e-3, max_passes=50, degree=3, gamma=None):
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.degree = degree
        self.gamma = gamma
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None
        self.K = None
        self.f = None
        self.loss_history = []
        self.test_accuracy_history = []
        self.eval_points = []
        self.iteration = 0

    def fit(self, X, y, X_test=None, y_test=None):
        m, n = X.shape
        self.X = X.astype(float)
        self.y = y.astype(float).reshape(-1, 1)
        self.alpha = np.zeros((m, 1))
        self.b = 0

        self.K = kernel_function(self.X, self.X, kernel_type=self.kernel,
                                 degree=self.degree, gamma=self.gamma)

        self.f = np.zeros((m, 1))

        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(m):
                E_i = self.f[i] - self.y[i]

                if (self.y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                        (self.y[i] * E_i > self.tol and self.alpha[i] > 0):

                    j = self._select_j(i, m)
                    E_j = self.f[j] - self.y[j]

                    alpha_i_old = self.alpha[i].copy()
                    alpha_j_old = self.alpha[j].copy()

                    if self.y[i] != self.y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    if L == H:
                        continue

                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue

                    self.alpha[j] -= self.y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])

                    b1 = self.b - E_i - \
                         self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] - \
                         self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]
                    b2 = self.b - E_j - \
                         self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] - \
                         self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    delta_alpha_i = self.alpha[i] - alpha_i_old
                    delta_alpha_j = self.alpha[j] - alpha_j_old
                    self.f += (delta_alpha_i * self.y[i] * self.K[:, i].reshape(-1, 1)) + \
                              (delta_alpha_j * self.y[j] * self.K[:, j].reshape(-1, 1))

                    num_changed_alphas += 1
                    self.iteration += 1

            if X_test is not None and y_test is not None:
                y_pred_test = self.predict(X_test)
                test_accuracy = self.evaluate_accuracy(y_test, y_pred_test)
                self.test_accuracy_history.append(test_accuracy)
                self.eval_points.append(self.iteration)

            loss = self._compute_loss()
            self.loss_history.append(loss)

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

    def _select_j(self, i, m):
        E_i = self.f[i] - self.y[i]
        E_diff = np.abs(self.f.flatten() - E_i.flatten())
        E_diff[i] = -1
        j = np.argmax(E_diff)
        if E_diff[j] == -1:
            j = i
            while j == i:
                j = np.random.randint(0, m)
        return j

    def _compute_loss(self):
        term1 = np.sum(self.alpha)
        term2 = 0.5 * np.sum(self.alpha * self.y * (self.K @ self.alpha * self.y))
        loss = term2 - term1
        return loss.item()

    def predict(self, X):
        K = kernel_function(X, self.X, kernel_type=self.kernel,
                            degree=self.degree, gamma=self.gamma)
        y_pred = (K @ (self.alpha * self.y)) + self.b
        return np.sign(y_pred)

    def evaluate_accuracy(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        return np.mean(y_true == y_pred)

import numpy as np
import matplotlib.pyplot as plt

from build_data import get_processed_data
from lin_regression import (
    transform_labels as transform_labels_lr,
    add_intercept as add_intercept_lr,
    ridge_regression,
    predict as predict_lr,
    evaluate_accuracy as evaluate_accuracy_lr
)
from lin_classification import (
    transform_labels as transform_labels_lc,
    add_intercept as add_intercept_lc,
    LinearClassifier,
)
from svm import SVM_SMO
from sklearn.model_selection import train_test_split

best_params_lc = None
best_params_svm = None
best_lambda_lr = None

def load_and_preprocess_data():
    X_train, X_val, y_train, y_val, test_df = get_processed_data()
    print(f'Размер тренировочной выборки: {X_train.shape}')
    print(f'Размер валидационной выборки: {X_val.shape}')

    X_full = np.vstack((X_train.values, X_val.values))
    y_full = np.hstack((y_train.values, y_val.values))
    return X_full, y_full

def prepare_data_for_regression(X_full, y_full):
    y_transformed = transform_labels_lr(y_full).astype(float)
    X_with_intercept = add_intercept_lr(X_full).astype(float)
    return X_with_intercept, y_transformed

def normalize_features(X):
    X_features = X[:, 1:]
    X_mean = np.mean(X_features, axis=0)
    X_std = np.std(X_features, axis=0)
    X_std_corrected = np.where(X_std == 0, 1, X_std)
    X_features_normalized = (X_features - X_mean) / X_std_corrected
    X_normalized = np.hstack((np.ones((X_features_normalized.shape[0], 1)), X_features_normalized))
    return X_normalized

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def linear_regression_with_ridge(X_train, y_train, X_test, y_test):
    lambda_values = np.logspace(-4, 4, 30)
    best_lambda = None
    best_accuracy = 0
    accuracies = []

    weights = []
    for lambda_reg in lambda_values:
        w = ridge_regression(X_train, y_train, lambda_reg)
        weights.append(w.flatten())
        y_pred = predict_lr(X_test, w)
        accuracy = evaluate_accuracy_lr(y_test, y_pred)
        accuracies.append(accuracy)
        print(f'Lambda: {lambda_reg:.4f}, Accuracy: {accuracy:.4f}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lambda = lambda_reg

    print(f'Лучшее значение lambda для гребневой регрессии: {best_lambda}, Accuracy: {best_accuracy:.4f}')

    w_best = ridge_regression(X_train, y_train, best_lambda)
    y_pred_test = predict_lr(X_test, w_best)
    final_accuracy = evaluate_accuracy_lr(y_test, y_pred_test)
    print(f'Итоговая точность на тестовой выборке (Линейная регрессия): {final_accuracy:.4f}')

    global best_lambda_lr
    best_lambda_lr = best_lambda

    plt.figure()
    plt.semilogx(lambda_values, accuracies, marker='o')
    plt.xlabel('Lambda (регуляризация)')
    plt.ylabel('Точность')
    plt.title('Зависимость точности линейной регрессии от регуляризации')
    plt.grid(True)
    plt.show()

    return w_best, final_accuracy, best_lambda

def linear_classification_with_gd(X_train, y_train, X_test, y_test):
    loss_functions = ['mse', 'logistic', 'exponential']
    learning_rates = [0.001, 0.005, 0.01]
    lambda1_values = [0.0, 0.01, 0.1]
    lambda2_values = [0.0, 0.01, 0.1]
    n_iterations = 1000

    best_params = {}
    best_accuracy = 0

    for loss in loss_functions:
        for lr in learning_rates:
            for l1 in lambda1_values:
                for l2 in lambda2_values:
                    classifier = LinearClassifier(
                        loss=loss,
                        learning_rate=lr,
                        n_iterations=n_iterations,
                        lambda1=l1,
                        lambda2=l2
                    )
                    classifier.fit(X_train, y_train, X_test, y_test)
                    y_pred = classifier.predict(X_test)
                    accuracy = classifier.evaluate_accuracy(y_test, y_pred)
                    print(f'Loss: {loss}, LR: {lr}, L1: {l1}, L2: {l2}, Accuracy: {accuracy:.4f}')
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            'loss': loss,
                            'learning_rate': lr,
                            'lambda1': l1,
                            'lambda2': l2,
                            'n_iterations': n_iterations
                        }
                        best_loss_history = classifier.test_loss_history.copy()

    print(f'Лучшие параметры для линейной классификации: {best_params}, Accuracy: {best_accuracy:.4f}')

    global best_params_lc
    best_params_lc = best_params

    classifier_best = LinearClassifier(
        loss=best_params['loss'],
        learning_rate=best_params['learning_rate'],
        n_iterations=best_params['n_iterations'],
        lambda1=best_params['lambda1'],
        lambda2=best_params['lambda2']
    )
    classifier_best.fit(X_train, y_train, X_test, y_test)
    y_pred_test = classifier_best.predict(X_test)
    final_accuracy = classifier_best.evaluate_accuracy(y_test, y_pred_test)
    print(f'Итоговая точность на тестовой выборке (Линейная классификация): {final_accuracy:.4f}')

    plot_loss_curve(classifier_best.test_loss_history, 'Кривая функции потерь на тестовом множестве (Линейная классификация)')
    return classifier_best, final_accuracy

def svm_with_smo(X_train, y_train, X_test, y_test):
    kernels = ['linear', 'polynomial', 'rbf']
    C_values = [0.01, 0.1, 1]
    tol_values = [1e-3, 1e-4]
    max_passes_values = [5, 10, 50, 100]
    degree_values = [2, 3, 4]
    gamma_values = [0.01, 0.1, 1.0]

    best_params = {}
    best_accuracy = 0

    for kernel in kernels:
        for C in C_values:
            for tol in tol_values:
                for max_passes in max_passes_values:
                    if kernel == 'polynomial':
                        for degree in degree_values:
                            svm = SVM_SMO(
                                kernel=kernel,
                                C=C,
                                tol=tol,
                                max_passes=max_passes,
                                degree=degree
                            )
                            svm.fit(X_train, y_train, X_test, y_test)
                            y_pred = svm.predict(X_test)
                            accuracy = svm.evaluate_accuracy(y_test, y_pred)
                            print(
                                f'Kernel: {kernel}, C: {C}, tol: {tol}, '
                                f'max_passes: {max_passes}, degree: {degree}, Accuracy: {accuracy:.4f}')
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_params = {
                                    'kernel': kernel,
                                    'C': C,
                                    'tol': tol,
                                    'max_passes': max_passes,
                                    'degree': degree
                                }
                    elif kernel == 'rbf':
                        for gamma in gamma_values:
                            svm = SVM_SMO(
                                kernel=kernel,
                                C=C,
                                tol=tol,
                                max_passes=max_passes,
                                gamma=gamma
                            )
                            svm.fit(X_train, y_train, X_test, y_test)
                            y_pred = svm.predict(X_test)
                            accuracy = svm.evaluate_accuracy(y_test, y_pred)
                            print(
                                f'Kernel: {kernel}, C: {C}, tol: {tol}, '
                                f'max_passes: {max_passes}, gamma: {gamma}, Accuracy: {accuracy:.4f}')
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_params = {
                                    'kernel': kernel,
                                    'C': C,
                                    'tol': tol,
                                    'max_passes': max_passes,
                                    'gamma': gamma
                                }
                    else:
                        svm = SVM_SMO(
                            kernel=kernel,
                            C=C,
                            tol=tol,
                            max_passes=max_passes
                        )
                        svm.fit(X_train, y_train, X_test, y_test)
                        y_pred = svm.predict(X_test)
                        accuracy = svm.evaluate_accuracy(y_test, y_pred)
                        print(
                            f'Kernel: {kernel}, C: {C}, tol: {tol}, '
                            f'max_passes: {max_passes}, Accuracy: {accuracy:.4f}')
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {
                                'kernel': kernel,
                                'C': C,
                                'tol': tol,
                                'max_passes': max_passes
                            }

    print(f'Лучшие параметры для SVM: {best_params}, Accuracy: {best_accuracy:.4f}')

    global best_params_svm
    best_params_svm = best_params

    svm_best = SVM_SMO(
        kernel=best_params['kernel'],
        C=best_params['C'],
        tol=best_params['tol'],
        max_passes=best_params['max_passes'],
        degree=best_params.get('degree', 3),
        gamma=best_params.get('gamma', None)
    )
    svm_best.fit(X_train, y_train, X_test, y_test)
    y_pred_test = svm_best.predict(X_test)
    final_accuracy = svm_best.evaluate_accuracy(y_test, y_pred_test)
    print(f'Итоговая точность на тестовой выборке (SVM): {final_accuracy:.4f}')

    plot_loss_curve(svm_best.loss_history, 'Кривая функции потерь на тестовом множестве (SVM)')

    return svm_best, final_accuracy

def collect_learning_curves(model_class, X_full, y_full, n_runs=50):
    test_accuracies = []
    eval_points_list = []

    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.5, stratify=y_full, random_state=run)

        if model_class == 'linear_classifier':
            classifier = LinearClassifier(
                loss=best_params_lc['loss'],
                learning_rate=best_params_lc['learning_rate'],
                n_iterations=best_params_lc['n_iterations'],
                lambda1=best_params_lc['lambda1'],
                lambda2=best_params_lc['lambda2'],
            )
            classifier.fit(X_train, y_train, X_test, y_test)
            test_accuracies.append(classifier.test_accuracy_history)
            eval_points_list.append(np.arange(0, best_params_lc['n_iterations']))
        elif model_class == 'svm':
            svm = SVM_SMO(
                kernel=best_params_svm['kernel'],
                C=best_params_svm['C'],
                tol=best_params_svm['tol'],
                max_passes=best_params_svm['max_passes'],
                degree=best_params_svm.get('degree', 3),
                gamma=best_params_svm.get('gamma', None)
            )
            svm.fit(X_train, y_train, X_test, y_test)
            test_accuracies.append(svm.test_accuracy_history)
            eval_points_list.append(svm.eval_points)

    max_length = max(len(acc) for acc in test_accuracies)
    test_accuracies_padded = [np.pad(acc, (0, max_length - len(acc)), 'edge') for acc in test_accuracies]
    eval_points_padded = [np.pad(points, (0, max_length - len(points)), 'edge') for points in eval_points_list]
    test_accuracies = np.array(test_accuracies_padded)
    eval_points = np.array(eval_points_padded).mean(axis=0)
    return test_accuracies, eval_points

def plot_loss_curve(loss_history, title):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(loss_history)), loss_history, label='Функция потерь')
    plt.xlabel('Итерация')
    plt.ylabel('Значение функции потерь')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_learning_curve_with_confidence(test_accuracies, eval_points, title, lr_accuracy):
    mean_accuracy = np.mean(test_accuracies, axis=0)
    std_accuracy = np.std(test_accuracies, axis=0)
    n_runs = test_accuracies.shape[0]
    standard_error = std_accuracy / np.sqrt(n_runs)
    confidence_interval = 1.96 * standard_error

    plt.figure(figsize=(8, 6))
    plt.plot(eval_points, mean_accuracy, label='Средняя точность на тестовом множестве')
    plt.fill_between(eval_points, mean_accuracy - confidence_interval, mean_accuracy + confidence_interval, alpha=0.2)
    plt.axhline(y=lr_accuracy, color='r', linestyle='--', label='Линейная регрессия')
    plt.xlabel('Итерация')
    plt.ylabel('Точность на тестовом множестве')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def get_linear_regression_accuracy(X_train_lr, y_train_lr, X_test_lr, y_test_lr):
    w = ridge_regression(X_train_lr, y_train_lr, best_lambda_lr)
    y_pred_test = predict_lr(X_test_lr, w)
    accuracy = evaluate_accuracy_lr(y_test_lr, y_pred_test)
    return accuracy

def main():
    X_full, y_full = load_and_preprocess_data()

    X, y = prepare_data_for_regression(X_full, y_full)
    X = normalize_features(X)

    X_train_lr, X_test_lr, y_train_lr, y_test_lr = split_data(X, y)

    print("\n=== Линейная регрессия с гребневой регуляризацией ===")
    w_lr, final_accuracy_lr, best_lambda_lr = linear_regression_with_ridge(X_train_lr, y_train_lr, X_test_lr, y_test_lr)

    print("\n=== Линейная классификация на основе градиентного спуска ===")
    classifier_lc, final_accuracy_lc = linear_classification_with_gd(X_train_lr, y_train_lr, X_test_lr, y_test_lr)

    X_svm = X[:, 1:]
    y_svm = y.flatten()

    print("\n=== Метод опорных векторов (SVM) с SMO ===")
    svm_model, final_accuracy_svm = svm_with_smo(X_train_lr[:, 1:], y_train_lr.flatten(), X_test_lr[:, 1:], y_test_lr.flatten())

    n_runs = 50
    test_accuracies_lc, eval_points_lc = collect_learning_curves('linear_classifier', X, y, n_runs)
    test_accuracies_svm, eval_points_svm = collect_learning_curves('svm', X_svm, y_svm, n_runs)

    lr_accuracy = get_linear_regression_accuracy(X_train_lr, y_train_lr, X_test_lr, y_test_lr)

    plot_learning_curve_with_confidence(
        test_accuracies_lc,
        eval_points_lc,
        'Кривая обучения для линейной классификации',
        lr_accuracy
    )

    plot_learning_curve_with_confidence(
        test_accuracies_svm,
        eval_points_svm,
        'Кривая обучения для SVM',
        lr_accuracy
    )

if __name__ == '__main__':
    main()

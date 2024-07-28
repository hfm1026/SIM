import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
from collections import defaultdict
from skopt import gp_minimize
from skopt.space import Real

class WeightedEnsemble:
    def __init__(self, save_model_dir, xgboost_step, adaboost_step, optimization_method='bayesian'):
        self.save_model_dir = save_model_dir
        self.xgboost_step = str(xgboost_step)
        self.adaboost_step = str(adaboost_step)
        self.optimization_method = optimization_method

    def predict(self):
        models = ['XGBoost', 'Adaboost']
        folds = range(1, 5)
        metrics = defaultdict(list)

        # Define the objective function for optimization
        def evaluate_auc(weight):
            weight = weight[0] if isinstance(weight, list) else weight
            fold_aucs = []
            for fold in folds:
                Xs_predict = np.load(f"{self.save_model_dir}/step0/X_test_fold{fold}.npy")
                y_true = np.load(f"{self.save_model_dir}/step0/y_test_fold{fold}.npy")
                fold_predictions = {}
                for model_name in models:
                    model_filename = f"{self.save_model_dir}/step{self.xgboost_step if model_name == 'XGBoost' else self.adaboost_step}/{model_name}_fold{fold}.joblib"
                    classifier = joblib.load(model_filename)
                    y_pred_proba = classifier.predict_proba(Xs_predict)[:, 1]
                    fold_predictions[model_name] = y_pred_proba
                ensemble_prediction = weight * fold_predictions['XGBoost'] + (1 - weight) * fold_predictions['Adaboost']
                fold_aucs.append(roc_auc_score(y_true, ensemble_prediction))
            return -np.mean(fold_aucs)

        # Optimization selector
        best_weight = 0.5  # Default in case of error
        if self.optimization_method == 'bayesian':
            res = gp_minimize(evaluate_auc, [Real(0.0, 1.0)], n_calls=50, random_state=0)
            best_weight = res.x[0]
        elif self.optimization_method == 'hill_climbing':
            best_weight, _ = self.hill_climbing_optimize(evaluate_auc, 0.5, 500)
        elif self.optimization_method == 'simulated_annealing':
            best_weight, _ = self.simulated_annealing_optimize(evaluate_auc, 0.5, 500, 1.0)

        # Recompute metrics using the best weight found
        self.recompute_metrics(best_weight, models, folds, metrics)
        return metrics

    def hill_climbing_optimize(self, func, initial_weight, n_iterations):
        current_weight = initial_weight
        current_value = func(current_weight)
        for _ in range(n_iterations):
            next_weight = current_weight + np.random.uniform(-0.1, 0.1)
            next_weight = np.clip(next_weight, 0, 1)
            next_value = func(next_weight)
            if next_value < current_value:
                current_weight, current_value = next_weight, next_value
        return current_weight, -current_value

    def simulated_annealing_optimize(self, func, initial_weight, n_iterations, temp):
        current_weight = initial_weight
        current_value = func(current_weight)
        for i in range(n_iterations):
            next_weight = current_weight + np.random.uniform(-0.1, 0.1)
            next_weight = np.clip(next_weight, 0, 1)
            next_value = func(next_weight)
            if next_value < current_value or np.exp((current_value - next_value) / temp) > np.random.rand():
                current_weight, current_value = next_weight, next_value
            temp *= 0.99  # Cooling
        return current_weight, -current_value

    def recompute_metrics(self, best_weight, models, folds, metrics):
        for fold in folds:
            Xs_predict = np.load(f"{self.save_model_dir}/step0/X_test_fold{fold}.npy")
            y_true = np.load(f"{self.save_model_dir}/step0/y_test_fold{fold}.npy")
            fold_predictions = {}
            for model_name in models:
                model_filename = f"{self.save_model_dir}/step{self.xgboost_step if model_name == 'XGBoost' else self.adaboost_step}/{model_name}_fold{fold}.joblib"
                classifier = joblib.load(model_filename)
                y_pred_proba = classifier.predict_proba(Xs_predict)[:, 1]
                fold_predictions[model_name] = y_pred_proba
            ensemble_prediction = best_weight * fold_predictions['XGBoost'] + (1 - best_weight) * fold_predictions['Adaboost']
            fold_auc = roc_auc_score(y_true, ensemble_prediction)
            fold_accuracy = accuracy_score(y_true, ensemble_prediction > 0.5)
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, ensemble_prediction)
            fold_auprc = auc(recall_curve, precision_curve)

            metrics['AUC'].append(fold_auc)
            metrics['Accuracy'].append(fold_accuracy)
            metrics['AUPRC'].append(fold_auprc)
        
        print("Best Weight: ", best_weight)
        print("Highest Average AUC Score across folds: ", np.mean(metrics['AUC']))
        print("Average Accuracy Score across folds: ", np.mean(metrics['Accuracy']))
        print("Average AUPRC Score across folds: ", np.mean(metrics['AUPRC']))
import pandas as pd
import numpy as np
import joblib
import os
import sklearn
from utils import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score,precision_recall_curve,auc
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore') 

class Train:
    def __init__(self,Xs,ys,cls_type,save_model_dir):
        self.models_dict = {
            'DT': DecisionTreeClassifier(),
            'KNN': KNeighborsClassifier(),
            'XGBoost': xgb.XGBClassifier(eval_metric='logloss',random_state=1),
            'Adaboost': AdaBoostClassifier(random_state=1)
        }
        self.Xs = Xs
        self.ys = ys
        self.cls_type = cls_type
        self.save_model_dir = save_model_dir

    def train_and_evaluate(self, model, X_train, y_train, X_test, y_test, smote_use=False):
        if smote_use:
            smote = SMOTE(random_state=1)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_score = auc(recall_curve,precision_curve)

        return acc, auc_score, tn, fp, fn, tp, recall, precision,pr_score

    def train(self,step):
        columns = ['Model', 'Fold', 'ACC', 'AUC', 'TN', 'FP', 'FN', 'TP', 'Recall', 'Precision','PR_score']
        results_df = pd.DataFrame(columns=columns)
        model_predictions = defaultdict(list)
        np.random.seed(1)
        skf = StratifiedKFold(n_splits=4)
        if step == 0:
            for fold, (train_index, test_index) in enumerate(skf.split(self.Xs, self.ys), 1):
                ensemble_fold_predictions = []
                X_train, X_test = self.Xs[train_index], self.Xs[test_index]
                y_train, y_test = self.ys[train_index], self.ys[test_index]
                np.save(os.path.join(create_dir(f"{self.save_model_dir}/step{step}"),f"X_train_fold{fold}.npy"), X_train)
                np.save(os.path.join(create_dir(f"{self.save_model_dir}/step{step}"),f"X_test_fold{fold}.npy"),X_test)
                np.save(os.path.join(create_dir(f"{self.save_model_dir}/step{step}"),f"y_train_fold{fold}.npy"),y_train)
                np.save(os.path.join(create_dir(f"{self.save_model_dir}/step{step}"),f"y_test_fold{fold}.npy"),y_test)
                for model_name, model in self.models_dict.items():
                    acc, auc_score, tn, fp, fn, tp, recall, precision,pr_score = self.train_and_evaluate(model, X_train, y_train, X_test, y_test)
                    results_df = pd.concat([results_df, pd.DataFrame([[model_name, fold, acc, auc_score, tn, fp, fn, tp, recall, precision,pr_score]], columns=columns)], ignore_index=True)
                    os.makedirs(self.save_model_dir, exist_ok=True)
                    joblib.dump(model, f"{self.save_model_dir}/step{step}/{model_name}_fold{fold}.joblib")



                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    model_predictions[model].append(y_pred_proba)
                    ensemble_fold_predictions.append(y_pred_proba)
                # Calculate the ensemble results of the current fold
                ensemble_prediction = np.mean(ensemble_fold_predictions, axis=0)
                model_predictions['Ensemble'].append(ensemble_prediction)
            for index in range(len(model_predictions['Ensemble'])):
                model_name = "ensemble"
                y_test = np.load(f"{self.save_model_dir}/step0/y_test_fold{index+1}.npy")
                y_pred_proba_ensemble = model_predictions['Ensemble'][index]
                y_pred_ensemble = np.where(y_pred_proba_ensemble > 0.5, 1, 0)
                acc = accuracy_score(y_test, y_pred_ensemble)
                auc_score = roc_auc_score(y_test, y_pred_proba_ensemble)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ensemble).ravel()
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba_ensemble)
                pr_score = sklearn.metrics.auc(recall_curve,precision_curve)
                results_df = pd.concat([results_df, pd.DataFrame([[model_name,index+1, acc, auc_score, tn, fp, fn, tp, recall, precision,pr_score]], columns=columns)], ignore_index=True)
        else:
            for fold in range(1,5):
                ensemble_fold_predictions = []
                X_train = np.load(f"{self.save_model_dir}/step{step}/X_train_fold{fold}.npy")
                X_test = np.load(f"{self.save_model_dir}/step0/X_test_fold{fold}.npy")
                y_train = np.load(f"{self.save_model_dir}/step{step}/y_train_fold{fold}.npy")
                y_test = np.load(f"{self.save_model_dir}/step0/y_test_fold{fold}.npy")
                for model_name, model in self.models_dict.items():
                    acc, auc_score, tn, fp, fn, tp, recall, precision,pr_score = self.train_and_evaluate(model, X_train, y_train, X_test, y_test)
                    results_df = pd.concat([results_df, pd.DataFrame([[model_name, fold, acc, auc_score, tn, fp, fn, tp, recall, precision,pr_score]], columns=columns)], ignore_index=True)
                    os.makedirs(self.save_model_dir, exist_ok=True)
                    joblib.dump(model, f"{self.save_model_dir}/step{step}/{model_name}_fold{fold}.joblib")



                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    model_predictions[model].append(y_pred_proba)
                    ensemble_fold_predictions.append(y_pred_proba)

                ensemble_prediction = np.mean(ensemble_fold_predictions, axis=0)
                model_predictions['Ensemble'].append(ensemble_prediction)
            for index in range(len(model_predictions['Ensemble'])):
                model_name = "ensemble"
                y_test = np.load(f"{self.save_model_dir}/step0/y_test_fold{index+1}.npy")
                y_pred_proba_ensemble = model_predictions['Ensemble'][index]
                y_pred_ensemble = np.where(y_pred_proba_ensemble > 0.5, 1, 0)
                acc = accuracy_score(y_test, y_pred_ensemble)
                auc_score = roc_auc_score(y_test, y_pred_proba_ensemble)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ensemble).ravel()
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba_ensemble)
                pr_score = sklearn.metrics.auc(recall_curve,precision_curve)
                results_df = pd.concat([results_df, pd.DataFrame([[model_name,index+1, acc, auc_score, tn, fp, fn, tp, recall, precision,pr_score]], columns=columns)], ignore_index=True)
                                    
        results_df = results_df.sort_values(["Model","Fold"]).reset_index(drop=True)
        return results_df

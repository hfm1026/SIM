import numpy as np
import pandas as pd
from collections import defaultdict
import joblib

class Predictor:
    def __init__(self,Xs_predict,sample_list,save_model_dir):
        self.save_model_dir = save_model_dir
        self.Xs_predict = Xs_predict
        self.sample_list = sample_list

    def predict(self,step):
        predictions_df = pd.DataFrame(self.sample_list, columns=['SampleName'])
        # Store the predictive outcomes for each model
        model_predictions = defaultdict(list)
        models = ['DT', 'KNN', 'XGBoost', 'Adaboost']
        folds = range(1, 5)
        # Load the model and make predictions
        for model in models:
            for fold in folds:
                model_filename = f"{self.save_model_dir}/step{step}/{model}_fold{fold}.joblib"
                classifier = joblib.load(model_filename)
                y_pred_proba = classifier.predict_proba(self.Xs_predict)
                y_pred_proba = y_pred_proba[:, 1]
                model_predictions[model].append(y_pred_proba)
        # Calculate the sum of the predicted results for each model
        for model, predictions in model_predictions.items():
            predictions_df[model] = np.mean(predictions, axis=0)
        predictions_df['ensembl_xgb_ada'] = predictions_df[['XGBoost',"Adaboost"]].mean(axis=1)
        predictions_df = predictions_df.sort_values("ensembl_xgb_ada",ascending=False).reset_index(drop=True)
        return predictions_df
    
    
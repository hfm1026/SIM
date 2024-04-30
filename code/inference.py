import os
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

cls_type = "target"
directory = f'../result/{cls_type}/ensemble_xgb_ada/metric'

data = {'ACC': [], 'AUC': [], 'Recall': [], 'Precision': [], 'PR_score': [], 'Step': []}

for filename in os.listdir(directory):
    if filename.startswith("train_metric_step") and filename.endswith(".csv"):
        step = int(filename.split("step")[1].split(".")[0])
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        adaboost_data = df[df['Model'] == 'XGBoost'].iloc[0]
        data['ACC'].append(adaboost_data['ACC'])
        data['AUC'].append(adaboost_data['AUC'])
        data['Recall'].append(adaboost_data['Recall'])
        data['Precision'].append(adaboost_data['Precision'])
        data['PR_score'].append(adaboost_data['PR_score'])
        data['Step'].append(step)

metrics_df = pd.DataFrame(data)
metrics_df.sort_values('Step', inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(metrics_df['Step'].values, metrics_df['ACC'].values, label='ACC')
plt.plot(metrics_df['Step'].values, metrics_df['AUC'].values, label='AUC')
plt.plot(metrics_df['Step'].values, metrics_df['Recall'].values, label='Recall')
plt.plot(metrics_df['Step'].values, metrics_df['Precision'].values, label='Precision')
plt.plot(metrics_df['Step'].values, metrics_df['PR_score'].values, label='PR_score')
plt.xlabel('Step')
plt.ylabel('Metrics')
plt.title('XGBboost Metrics Over Steps')
plt.legend()
plt.show()


cls_type = "target"
# step 49
predictions_best_adaboost = pd.read_csv(f"../result/{cls_type}/ensemble_xgb_ada/infer/inference_df_step49.csv")
# step 35
predictions_best_xgboost = pd.read_csv(f"../result/{cls_type}/ensemble_xgb_ada/infer/inference_df_step35.csv")

prediction_ensemble_target = pd.merge(predictions_best_adaboost[['SampleName','Adaboost']], predictions_best_xgboost[['SampleName','XGBoost']], on='SampleName')
prediction_ensemble_target['Ensemble'] = prediction_ensemble_target[['Adaboost','XGBoost']].mean(axis=1)
prediction_ensemble_target_fiter = prediction_ensemble_target[prediction_ensemble_target['Ensemble'] > 0.9]
print("all postive sample number: ",len(prediction_ensemble_target_fiter))
print('-' * 80)

cls_type = "temp"
# step 63
predictions_best_adaboost = pd.read_csv(f"../result/{cls_type}/ensemble_xgb_ada/infer/inference_df_step63.csv")
# step 69
predictions_best_xgboost = pd.read_csv(f"../result/{cls_type}/ensemble_xgb_ada/infer/inference_df_step69.csv")

prediction_ensemble_temp = pd.merge(predictions_best_adaboost[['SampleName','Adaboost']], predictions_best_xgboost[['SampleName','XGBoost']], on='SampleName')
prediction_ensemble_temp['Ensemble'] = prediction_ensemble_temp[['Adaboost','XGBoost']].mean(axis=1)
prediction_ensemble_temp_fiter = prediction_ensemble_temp[prediction_ensemble_temp['Ensemble'] > 0.5]
print("all postive sample number: ",len(prediction_ensemble_temp_fiter))

# make inetersection
temp_set = set(prediction_ensemble_temp_fiter['SampleName'].tolist())
temp_set = {filename.replace('.fa', '') for filename in temp_set}
target_set = set(prediction_ensemble_target_fiter['SampleName'].tolist())
target_set = {filename.replace('.pdb', '') for filename in target_set}
intersection = target_set.intersection(temp_set)
print("The number of RNA-targeting and room-temperature operating Agos is: ",len(intersection))

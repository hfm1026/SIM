import pandas as pd
import functools
import numpy as np
import os

def get_structure_feature_single(file_name,cls):
    if cls == "target":
        feature_matrix = np.load(f"{file_name}.npy")
        feature_matrix_mean = np.mean(feature_matrix, axis=1)
        return feature_matrix_mean
    else:
        feature_matrix = np.load(f"{file_name}.npy")
        return feature_matrix   
         
def add_sample(predictions_df,index,sample_new,add_sample_path,cls_type):
    add_sample = predictions_df['SampleName'].iloc[index]
    if add_sample not in sample_new:
        sample_new.append(add_sample)
    else:
        new_ranked_list = predictions_df['SampleName']
        deduplicated_list = [item for item in new_ranked_list if item not in sample_new]
        add_sample = deduplicated_list[index]
        sample_new.append(add_sample)
    add_sample_structure = os.path.join(add_sample_path,add_sample)
    aver_df = get_structure_feature_single(add_sample_structure,cls_type)
    return aver_df,sample_new

def add_sample_balance(predictions_df,threshold,sample_new,add_sample_path,cls_type):
    add_sample_pos = predictions_df['SampleName'].iloc[0]
    add_sample_neg = predictions_df['SampleName'].iloc[-1]
    if add_sample_pos not in sample_new:
        sample_new.append(add_sample_pos)
    else:
        new_ranked_list = predictions_df['SampleName']
        deduplicated_list = [item for item in new_ranked_list if item not in sample_new]
        add_sample_pos = deduplicated_list[0]
        cls_probability = predictions_df[predictions_df["SampleName"] == add_sample_pos]['ensembl_xgb_ada'].values[0]
        if cls_probability >= threshold:
            sample_new.append(add_sample_pos)
        else:
            raise Exception(f"Positive sample {add_sample_pos} not in threshold ({threshold}), which {add_sample_pos} classification probability is {cls_probability}")
                            
    if add_sample_neg not in sample_new:
        sample_new.append(add_sample_neg)
    else:
        new_ranked_list = predictions_df['SampleName']
        deduplicated_list = [item for item in new_ranked_list if item not in sample_new]
        add_sample_neg = deduplicated_list[-1]
        cls_probability = predictions_df[predictions_df["SampleName"] == add_sample_neg]['ensembl_xgb_ada'].values[0]
        if cls_probability <= threshold:
            sample_new.append(add_sample_neg)
        else:
            raise Exception(f"Negative sample {add_sample_neg} not in threshold ({threshold}), which {add_sample_neg} classification probability is {cls_probability}")
    add_sample_pos_path = os.path.join(add_sample_path,add_sample_pos)
    add_sample_neg_path = os.path.join(add_sample_path,add_sample_neg)
    aver_df_pos = get_structure_feature_single(add_sample_pos_path,cls_type)
    aver_df_neg = get_structure_feature_single(add_sample_neg_path,cls_type)
    aver_df = np.vstack((aver_df_pos, aver_df_neg))
    return aver_df,sample_new

def update_single_sample(save_step_dir,x_train,y_train,aver_df,fold,label):
    if not os.path.exists(save_step_dir):
        os.makedirs(save_step_dir)
    updated_x_train = np.vstack((x_train, aver_df))
    np.save(os.path.join(save_step_dir,f"X_train_fold{fold}.npy"),updated_x_train)
    updated_y_train = np.append(y_train, label)
    np.save(os.path.join(save_step_dir,f"y_train_fold{fold}.npy"),updated_y_train)

def update_double_sample(save_step_dir,x_train,y_train,aver_df,fold):
    if not os.path.exists(save_step_dir):
        os.makedirs(save_step_dir)
    updated_x_train = np.vstack((x_train, aver_df))
    np.save(os.path.join(save_step_dir,f"X_train_fold{fold}.npy"),updated_x_train)
    updated_y_train = np.append(y_train, 1)
    updated_y_train = np.append(updated_y_train, 0)
    np.save(os.path.join(save_step_dir,f"y_train_fold{fold}.npy"),updated_y_train)      

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
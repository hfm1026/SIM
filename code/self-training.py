import numpy as np
from trainer import *
from predictor import *
from utils import *
import argparse
import os
import random

parser = argparse.ArgumentParser(description='Self-training for Ago')
parser.add_argument("--dir", type=str, default="/home/hfm/project/argo_extract/AGO_summary_structure/self_training", required=True,
                    help='')
parser.add_argument("--cls", type=str, default="target", required=True,
                    help='')
parser.add_argument('--cls_threshold', type=int, default=0.5, metavar='N', required=False,
                    help='')
parser.add_argument("--save_model_dir", type=str, default='./data', required=True,
                    help='')
parser.add_argument('--nfold', type=int, default=4, metavar='N',
                    help='')
parser.add_argument('--total_step', type=int, default=200, metavar='N',
                    help='')
args = parser.parse_args()

def main(args):
    sample_new = []
    for step in range(args.total_step):
        create_dir(args.save_model_dir)
        create_dir(f"{args.save_model_dir}/metric")
        create_dir(f"{args.save_model_dir}/infer")
        save_test_result = f"{args.save_model_dir}/metric/train_metric_step{step}.csv"
        save_predict_result = f"{args.save_model_dir}/metric/predictions_df_step{step}.csv"
        save_model_step = create_dir(f"{args.save_model_dir}/step{step}/")
        save_step_dir = create_dir(f"{args.save_model_dir}/step{step+1}/")
        save_inference_dir = f"{args.save_model_dir}/infer/inference_df_step{step}.csv"
        # train
        results_df = trainer.train(step)
        grouped_df = results_df.groupby('Model').mean()
        grouped_df = grouped_df.reset_index(drop=False)
        grouped_df.to_csv(save_test_result,index=False)
        # predict
        predictions_df = predictor_short.predict(step)
        predictions_df.to_csv(save_predict_result,index=False)
        # inference
        predictions_df_long = predictor_long.predict(step)
        predictions_df_long.to_csv(save_inference_dir,index=False)
        # create new sample feature
        for fold in range(1,args.nfold+1):
            X_train_step = np.load(f"{save_model_step}/X_train_fold{fold}.npy")
            y_train_step = np.load(f"{save_model_step}/y_train_fold{fold}.npy")
            add_sample_path = f'{args.dir}/feature/{args.cls}/unlabel_short/'
            positive_count = np.count_nonzero(y_train_step)
            negative_count = y_train_step.size - positive_count
            print("Step: ",step,"\t","Fold: ",fold,"\n")
            print("Postive: ",positive_count,"\t","Negative: ",negative_count,"\n")

            # If the number of positive samples is less than the number of negative samples, then add one positive sample.
            if positive_count < negative_count:
                index = 0
                label = 1
                aver_df,sample_new = add_sample(predictions_df,index,sample_new,add_sample_path,args.cls)
                update_single_sample(save_step_dir,X_train_step,y_train_step,aver_df,fold,label)

            # If the number of positive samples is greater than the number of negative samples, then add a negative sample.     
            elif positive_count > negative_count:
                index = -1
                label = 0
                aver_df,sample_new = add_sample(predictions_df,index,sample_new,add_sample_path,args.cls)
                update_single_sample(save_step_dir,X_train_step,y_train_step,aver_df,fold,label)      

            # If the number of positive samples is equal to the number of negative samples, then add one positive sample and one negative sample.
            else:
                aver_df,sample_new = add_sample_balance(predictions_df,args.cls_threshold,sample_new,add_sample_path,args.cls)  
                update_double_sample(save_step_dir,X_train_step,y_train_step,aver_df,fold)   
        print("Step: ",step,"\t","Retrain sample numbers: ",len(sample_new),"\n")
        



if __name__ == '__main__':
    Xs = np.load(f'{args.dir}/feature/{args.cls}/label/Xs_remove_postdoc3_RsAgo_add_new.npy')
    ys = np.load(f'{args.dir}/feature/{args.cls}/label/ys_remove_postdoc3_RsAgo_add_new.npy')
    Xs_short = np.load(f'{args.dir}/feature/{args.cls}/unlabel_short_combine/Xs_predict.npy')
    name_list_short = sorted(os.listdir(f'{args.dir}/feature/{args.cls}/unlabel_short/'))
    name_list_short = [s.split(".npy")[0] for s in name_list_short if s.endswith(".npy")]
    Xs_long = np.load(f'{args.dir}/feature/{args.cls}/unlabel_long_combine/Xs_predict.npy')
    name_list_long = sorted(os.listdir(f'{args.dir}/feature/{args.cls}/unlabel_long/'))
    name_list_long = [s.split(".npy")[0] for s in name_list_long if s.endswith(".npy")]    

    trainer = Train(Xs,ys,args.cls,args.save_model_dir)
    predictor_short = Predictor(Xs_short,name_list_short,args.save_model_dir)
    predictor_long = Predictor(Xs_long,name_list_long,args.save_model_dir)
    main(args)

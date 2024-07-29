# Self-Iterative Model (SIM)
Advancing RNA-Targeting Argonaute Discovery with Self-Iterative AI Model from Scarce Experimental Data

Here we provide code to implement SIM for predicting RNA-targeting and room-temperature operating Agos.

Please cite:

## Repo contents
There are several main components of the repo.
- `data`: The information of labeled and unlabeled Agos and protein structure.
- `code`: The python code used to implement the SIM and extract feature from Agos.
- `result`: The baseline models and data in each iterative step.
- `feature`: The feature of Agos in nucleic acid targeting and operating temperature classification task.
- `environment.yml`: Software dependencies for conda environment.

## Train & Evaluation

1. Create the conda environment from the environment.yml file:
```
    conda env create -f environment.yml
```

2. Activate the new conda environment:
```
    conda activate SIM
```

3. run self-traning.py with following shell code:
```
for cls_type in temp target
do
    python ./code/self-training.py --dir ~/SIM \
                            --cls $cls_type \
                            --save_model_dir ~/SIM/result/$cls_type/ensemble_xgb_ada \
                            --nfold 4 \
                            --total_step 200 
done
```

4. The results of model and prediction accuracy are saved in `result`. The model performances of test data are saved in `metric` file  and long Agos predict probability is saved in `inference` file. The model and data for each iterative step are saved in file start with `step`. The `metrics.xlsx` in `result` file  is the filter results of pLDDT, probability of protein solubility, and structure similarity.

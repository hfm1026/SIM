for class in "target" "temp"
do
    python ./code/self-training.py --dir ~/SIM \
                            --cls ${class} \
                            --save_model_dir ~SIM/result/${class}/ \
                            --nfold 4 \
                            --total_step 200 
done
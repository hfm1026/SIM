import pandas as pd
import numpy as np
import iFeatureOmegaCLI
import os

feature = "APAAC"
def get_sequence_feature(feature_name, protein_file):
    protein = iFeatureOmegaCLI.iProtein(protein_file)
    status = protein.get_descriptor(feature_name)
    feature_matrix = protein.encodings.values
    return feature_matrix

fasta_datapath = "~/SIM/data/Labeled_Data_Sequence"
fasta_file_list = os.listdir(fasta_datapath)
for fasta_name in fasta_file_list:
    fasta_name_prefix = fasta_name.split(".fasta")[0]
    fasta_path = os.path.join(fasta_datapath,fasta_name)
    feature_matrix = get_sequence_feature(feature,fasta_path)
    save_path = f'~/SIM/feature/temp/label_single_all/{feature}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path,f'{fasta_name_prefix}.npy'), feature_matrix)
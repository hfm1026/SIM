import pandas as pd
import numpy as np
import iFeatureOmegaCLI
import os

cls_list = ["Target_cls", "Temp_cls"]
feature = "APAAC"
def get_sequence_feature(feature_name, protein_file):
    protein = iFeatureOmegaCLI.iProtein(protein_file)
    status = protein.get_descriptor(feature_name)
    feature_matrix = protein.encodings.values
    return feature_matrix

fasta_datapath = "/home/hfm/project/argo_extract/AGO_sequence/ago/single_fasta/unlabel_short"
fasta_file_list = os.listdir(fasta_datapath)
for fasta_name in fasta_file_list:
    fasta_name_prefix = fasta_name.split(".fasta")[0]
    fasta_path = os.path.join(fasta_datapath,fasta_name)
    feature_matrix = get_sequence_feature(feature,fasta_path)
    save_path = f'/home/hfm/project/argo_extract/AGO_sequence/ago/single_feature/unlabel_short/{feature}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path,f'{fasta_name_prefix}.npy'), feature_matrix)
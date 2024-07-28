import pandas as pd
import iFeatureOmegaCLI
import os
import numpy as np
from multiprocessing import Pool

strucutre_dir = "~/SIM/data/Labeled_Data_Structure"
feature_dir = "~/SIM/feature/target/label_single_all"
#os.makedirs(strucutre_dir, exist_ok=True)
os.makedirs(feature_dir, exist_ok=True)
# get structure feature
pdb_name_list = os.listdir(strucutre_dir)
def get_structure_feature(file_name):
    protein_structure = iFeatureOmegaCLI.iStructure(os.path.join(strucutre_dir, file_name))
    status = protein_structure.get_descriptor("AC_type2")
    feature_matrix = protein_structure.encodings
    feature_matrix = feature_matrix.T.values

    # Save each feature to a separate file
    np_file_name = file_name.rsplit('.pdb')[0]
    np.save(os.path.join(feature_dir, np_file_name), feature_matrix)

    return feature_matrix

def process_files_parallel(file_names):
    with Pool(os.cpu_count()) as pool:
        pool.map(get_structure_feature, file_names)

# Process files in parallel and save each file's features separately
process_files_parallel(pdb_name_list)
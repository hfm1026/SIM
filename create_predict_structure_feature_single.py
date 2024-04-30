import pandas as pd
import iFeatureOmegaCLI
import os
import numpy as np
from multiprocessing import Pool

# get structure feature
pdb_name_list = os.listdir("./predict_structure")
def get_structure_feature(file_name):
    protein_structure = iFeatureOmegaCLI.iStructure(f"./predict_structure/{file_name}")
    status = protein_structure.get_descriptor("AC_type2")
    feature_matrix = protein_structure.encodings
    feature_matrix = feature_matrix.T.values

    # Save each feature to a separate file
    np_file_name = file_name.rsplit('.pdb')[0]
    np.save(f"./predict_data_structure/feature_from_ifeature_single/{np_file_name}", feature_matrix)

    return feature_matrix

def process_files_parallel(file_names):
    with Pool(os.cpu_count()) as pool:
        pool.map(get_structure_feature, file_names)

# Process files in parallel and save each file's features separately
process_files_parallel(pdb_name_list)
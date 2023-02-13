import numpy as np
from lib.cloud_classification_data import get_scene, get_scene_mask, import_scene_names, unzeropad
from lib.quantum_classifiers import TA_hybrid_SVM, confusion_values, accuracy_from_confusion_values
from pathlib import Path
import pandas as pd
import glob
from sklearn.decomposition import PCA
from lib.circuits import WSWS

from pennylane import numpy as qnp
import pennylane as qml

def normalize_data(data, return_minmax = False):
    max = data.max(axis=0)
    min = data.min(axis=0)
    spread = max-min
    new_data = (data-min)/spread
    if(return_minmax):
        return(new_data, min, max)
    else:
        return(new_data)

def prepare_data(x_train, x_val, n_pca):
    x_train_norm, data_min, data_max = normalize_data(x_train, return_minmax=True)
    x_validate_norm = normalize_data(x_val)

    pca = PCA(n_components=n_pca)
    pca.fit(x_train_norm)
    
    x_train_pca = pca.transform(x_train_norm)
    x_validate_pca = pca.transform(x_validate_norm)

    x_train, pca_min, pca_max = normalize_data(x_train_pca, return_minmax=True)
    x_validate = normalize_data(x_validate_pca)
    return(x_train, x_validate, pca, data_min, data_max, pca_min, pca_max)
    

def simulate_scene(file_id, n_pca=4):
    base_path = Path('../38-Cloud_test')
    scenes_path = base_path/'test_sceneids_38-Cloud.csv'
    scene_names = import_scene_names(scenes_path)

    train_path = 'experiment_tr_data/*train*'
    train_files = sorted(glob.glob(train_path))
    validate_files = [s.replace('train', 'validate') for s in train_files]
    C_range = (0.01,150,10)

    print('---------------------------------')
    print(file_id+1, ' of ', len(train_files))
    train = pd.read_csv(train_files[file_id]).to_numpy()
    x_train = train[:,:-1]
    y_train = train[:,-1]

    validate = pd.read_csv(validate_files[file_id]).to_numpy()
    x_validate = validate[:,:-1]
    y_validate = validate[:,-1]

    x_train, x_validate, pca, data_min, data_max, pca_min, pca_max = prepare_data(x_train, x_validate,n_pca)
    
    num_of_W_layers = 2
    thetas = 2 * np.pi * qnp.random.random(size=(num_of_W_layers * x_train.shape[1], 3), requires_grad=True)
    clf = TA_hybrid_SVM(WSWS, thetas, pca=pca, data_min=data_min, data_max=data_max, pca_min=pca_min, pca_max=pca_max, classifier_name='WSWS')
    clf.validate(x_train, y_train, x_validate, y_validate, C_range, info = True, max_optimizer_steps=30, n_initializations=20)

    for scene_id in range(len(scene_names)):
        print('------------------------------')
        print('scene: ', scene_names[scene_id])
        print('------------------------------')
        scene_data, r_max, c_max, patches_ids_array, scene_names = get_scene(base_path, scene_id)
        gt_path = Path('../38-Cloud_test/Entire_scene_gts')
        gt_scene = get_scene_mask(gt_path, scene_names, scene_id)
        pred = clf.predict_scene_sp(scene_data=scene_data, r_max=r_max, c_max=c_max, array_of_patches_indices=patches_ids_array, scene_name=scene_names[scene_id])

        trimmed_pr_gt = unzeropad(pred,gt_scene)

        conf_val = confusion_values(trimmed_pr_gt, gt_scene)
        clf.export_scene_prediction_results(confusion_values=conf_val, dataset_name = 'SP_pca4')
        print('Finished prediction! Scene: ', scene_names[scene_id])
        print('Scene accurancy: ', accuracy_from_confusion_values(conf_val=conf_val))

import concurrent.futures

train_path = 'experiment_tr_data/*train*'
train_files = sorted(glob.glob(train_path))

N_list = range(len(train_files))

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(simulate_scene, N_list)
#simulate_scene(0)
from lib.classifiers import RBF_SVM, confusion_values, accuracy_from_confusion_values
import numpy as np
from lib.cloud_classification_data import get_scene, get_scene_mask, import_scene_names, unzeropad
from pathlib import Path
import pandas as pd
import glob


def simulate_scene(file_id):
    base_path = Path('../38-Cloud_test')
    scenes_path = base_path/'test_sceneids_38-Cloud.csv'
    scene_names = import_scene_names(scenes_path)


    train_path = 'experiment_tr_data/*train*'
    train_files = sorted(glob.glob(train_path))
    validate_files = [s.replace('train', 'validate') for s in train_files]

    gamma_range = (0.1,150,3)
    C_range = (0.1,150,3)

    print('---------------------------------')
    print(file_id+1, ' of ', len(train_files))
    train = pd.read_csv(train_files[file_id]).to_numpy()
    x_train = train[:,:-1]
    y_train = train[:,-1]

    validate = pd.read_csv(validate_files[file_id]).to_numpy()
    x_validate = validate[:,:-1]
    y_validate = validate[:,-1]

    rbf = RBF_SVM(gamma = 1)
    rbf.validate(x_train, y_train, x_validate, y_validate, C_range, gamma_range, info=True)

    for scene_id in range(len(scene_names)):
        scene_data, r_max, c_max, patches_ids_array, scene_names = get_scene(base_path, scene_id)
        gt_path = Path('../38-Cloud_test/Entire_scene_gts')
        gt_scene = get_scene_mask(gt_path, scene_names, scene_id)
        pred = rbf.predict_scene_sp(scene_data=scene_data, r_max=r_max, c_max=c_max, array_of_patches_indices=patches_ids_array, scene_name=scene_names[scene_id])

        trimmed_pr_gt = unzeropad(pred,gt_scene)

        conf_val = confusion_values(trimmed_pr_gt, gt_scene)
        rbf.export_scene_prediction_results(confusion_values=conf_val, dataset_name = 'SP')
        print('Finished prediction! Scene: ', scene_names[scene_id])
        print('Scene accurancy: ', accuracy_from_confusion_values(conf_val=conf_val))

import concurrent.futures

train_path = 'experiment_training_data/*train*'
train_files = sorted(glob.glob(train_path))

N_list = range(len(train_files))

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(simulate_scene, N_list)
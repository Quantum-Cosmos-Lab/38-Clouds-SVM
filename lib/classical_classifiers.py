from sklearn.svm import SVC
import numpy as np
import datetime
import pandas as pd
from scipy.stats import iqr
from skimage.segmentation import slic


#---------------------------------------------------------------------

class RBF_SVM:
    def __init__(self, gamma, classifier_name='rbf', C=1, pca=None, data_min = 0, data_max = 1, pca_min = 0, pca_max = 1):
        self.classifier_name = classifier_name

        self.clf = SVC(kernel = 'rbf', gamma=gamma, C=C)
        self.C = C
        self.gamma = gamma

        self.pca = pca
        self.data_min = data_min #array of minimal values for each feature
        self.data_max = data_max #array of maximal values for each feature
        self.pca_min = pca_min
        self.pca_max = pca_max

        self.trained = False
        self.training_acc = None
        self.N_train = None
        self.SV_n = None
        self.prediction_conf_values = None

        #For scene prediction
        self.scene_name = None
        self.confusion_values = None

    def train(self, x_train, y_train, info=True):
        self.clf.fit(x_train, y_train)
        if(self.clf.fit_status_):
            if(info):print("Not fitted")
            return(0)
        else:
            if(info):print("Fitted correctly")
            acc=self.clf.score(x_train, y_train)
            if(info):print("Fit accuracy: ", acc)
            self.N_train = x_train.shape[0]
            self.trained = True
            self.training_acc = acc
            self.SV_n = self.clf.n_support_
            return(acc)
                
    def validate(self, x_train, y_train, x_val, y_val, C_range, gamma_range, info=False):

        gammas = np.arange(gamma_range[0],gamma_range[1],gamma_range[2])
        Cs = np.arange(C_range[0],C_range[1],C_range[2])

        best_gamma = 0
        best_C = 0
        best_acc = 0

        for C in Cs:
            for gamma in gammas:
                self.change_params(C=C, gamma=gamma)
                self.train(x_train,y_train, info = False)
                pred = self.predict(x_val)
                acc = accuracy_from_confusion_values(confusion_values(pred, y_val))
                if(acc > best_acc):
                    best_acc = acc
                    best_gamma = gamma
                    best_C = C

        if(info):print('Best validation score: ', np.round(100*best_acc, decimals=2),'%')

        if(info):print('best gamma: ', best_gamma)
        if(info):print('best C: ', best_C)
        self.change_params(C=best_C, gamma=best_gamma)
        self.train(x_train,y_train, info = False)
        return(best_acc)

    def predict(self, x_test, y_test=None):
        prediction = self.clf.predict(x_test)
        if(np.any(y_test!=None)):
            self.prediction_conf_values = confusion_values(prediction, y_test)
        return(prediction)

    def predict_scene(self, scene_data, r_max, c_max, array_of_patches_indices, scene_name=None, patch_x_size = 384, patch_y_size = 384):
        self.scene_name=scene_name

        #print('Predicting scene')
        margin_point = np.zeros((1, 4))#n_wires=4#Prepare in advance a margin state as it is extremely frequent in the data
        #margin_state = get_states(margin_point, w, V1, V2)
        for j in range(r_max):#run through all rows of the scene
            if(j%1==0): print('N:', self.N_train,'| row number: ',j)

            for i in range(c_max):#run through all columns of the scene
                #print('col number: ', i)

                patch_id = array_of_patches_indices[j,i]#identify the patch_id of the patch in current element of the scene grid
                patch_p = scene_data.open_as_points(patch_id, include_nir=True)

                if(patch_p.sum()==0):#if the patch is empty then skip it with no clouds
                    predicted_array = np.zeros((patch_x_size, patch_y_size), dtype=int)
                else:
                    predicted_array = np.empty(patch_p.shape[0], dtype=int)

                    all_ids = np.arange(patch_p.shape[0])
                    margin_ids = np.where((patch_p == margin_point).all(axis=1))[0]
                    nonmargin_ids = np.delete(all_ids, margin_ids)

                    predicted_array[margin_ids] = 0

                    for row_n in nonmargin_ids:
                        test_row = patch_p[row_n].reshape(1,-1)
                        #states_test = get_states(test_row, w, V1, V2)
                        #ker_test = kernel_test(SV, states_test)
                        predicted_array[row_n] = self.clf.predict(test_row)[0]

                    predicted_array = predicted_array.reshape(patch_x_size, patch_y_size)
                if(i==0): 
                    row = predicted_array
                else:
                    row = np.concatenate((row,predicted_array),axis=1)
            if(j==0):
                predicted_gt = row
            else:
                predicted_gt = np.concatenate((predicted_gt,row),axis=0)
        return(predicted_gt)

    def predict_scene_sp(self, scene_data, r_max, c_max, array_of_patches_indices, numSegments = 200, sigma = 5, scene_name=None, patch_x_size = 384, patch_y_size = 384):
        self.scene_name=scene_name

        #print('Predicting scene')
        margin_point = np.zeros((1, 24))#n_wires=4#Prepare in advance a margin state as it is extremely frequent in the data
        #margin_state = get_states(margin_point, w, V1, V2)
        for j in range(r_max):#run through all rows of the scene
            if(j%1==0): print('N:', self.N_train,'| row number: ',j)

            for i in range(c_max):#run through all columns of the scene
                #print('col number: ', i)

                patch_id = array_of_patches_indices[j,i]#identify the patch_id of the patch in current element of the scene grid
                
                #patch_p = scene_data.open_as_points(patch_id, include_nir=True)

                image = scene_data.open_as_array(idx=patch_id)
                image_nir = scene_data.open_as_array(idx=patch_id, include_nir=True)

                predicted_array = np.empty((patch_x_size, patch_y_size), dtype=int)
                segments = slic(image, n_segments = numSegments, sigma = sigma, convert2lab=True, channel_axis=2)
                for segment_id in range(1,len(np.unique(segments))+1):
                    segment_mask = (segments == segment_id)#Create segment mask
                    segment = image_nir[segment_mask]#Create segment from the patch image
                    segment_gt = np.zeros((segment.shape[0]))#Create gt segment from patch gt
                    superpixel, feature_names= summarize_segment(segment=segment, segment_gt=segment_gt) # summarize_segment should be changed to take segment_gt as an optional argument
                    superpixel = superpixel[:-1].reshape(1,-1)
                    superpixel = (superpixel-self.data_min)/(self.data_max-self.data_min)#Normalize before PCA
                    if(self.pca!=None): 
                        superpixel = self.pca.transform(superpixel)
                        superpixel = (superpixel-self.pca_min)/(self.pca_max-self.pca_min)#Normalize after PCA
                    prediction = self.clf.predict(superpixel)[0]
                    #print(prediction)
                    predicted_array[segment_mask] = prediction

                if(i==0): 
                    row = predicted_array
                else:
                    row = np.concatenate((row,predicted_array),axis=1)
            if(j==0):
                predicted_gt = row
            else:
                predicted_gt = np.concatenate((predicted_gt,row),axis=0)
        return(predicted_gt)

    def change_params(self, C, gamma):
        self.C = C
        self.gamma = gamma
        self.clf = SVC(kernel='rbf', C=C, gamma=gamma)

    def export_scene_prediction_results(self, confusion_values, dataset_name = 'SVM_XM'):
        colnames = ['N', 'tp', 'tn', 'fp', 'fn', 'SV0', 'SV1', 'C', 'gamma', 'timestamp', 'scene']
        result = np.array([[self.N_train, confusion_values[0], confusion_values[1], confusion_values[2], confusion_values[3], self.SV_n[0], self.SV_n[1], self.C, self.gamma, get_timestamp(), self.scene_name]])
        result_df = pd.DataFrame(result, columns = colnames)
        result_df.to_csv('result_'+ self.classifier_name + '_' + dataset_name +'.csv', mode = 'a', index=False, header=False)

#---------------------------------------------------------------------

class Linear_SVM:
    def __init__(self, classifier_name='linear', C=1, pca=None, data_min = 0, data_max = 1, pca_min = 0, pca_max = 1):
        self.classifier_name = classifier_name

        self.clf = SVC(kernel = 'linear', C=C)
        self.C = C

        self.pca = pca
        self.data_min = data_min #array of minimal values for each feature
        self.data_max = data_max #array of maximal values for each feature
        self.pca_min = pca_min
        self.pca_max = pca_max

        self.trained = False
        self.training_acc = None
        self.N_train = None
        self.SV_n = None
        self.prediction_conf_values = None

        #For scene prediction
        self.scene_name = None
        self.confusion_values = None

    def train(self, x_train, y_train, info=True):
        self.clf.fit(x_train, y_train)
        if(self.clf.fit_status_):
            if(info):print("Not fitted")
            return(0)
        else:
            if(info):print("Fitted correctly")
            acc=self.clf.score(x_train, y_train)
            if(info):print("Fit accuracy: ", acc)
            self.N_train = x_train.shape[0]
            self.trained = True
            self.training_acc = acc
            self.SV_n = self.clf.n_support_
            return(acc)
                
    def validate(self, x_train, y_train, x_val, y_val, C_range, info=False):

        Cs = np.arange(C_range[0],C_range[1],C_range[2])

        best_C = 0
        best_acc = 0

        for C in Cs:
            self.change_params(C=C)
            self.train(x_train,y_train, info = False)
            pred = self.predict(x_val)
            acc = accuracy_from_confusion_values(confusion_values(pred, y_val))
            if(acc > best_acc):
                best_acc = acc
                best_C = C

        if(info):print('Best validation score: ', np.round(100*best_acc, decimals=2),'%')

        if(info):print('best C: ', best_C)
        self.change_params(C=best_C)
        self.train(x_train,y_train, info = False)
        return(best_acc)

    def predict(self, x_test, y_test=None):
        prediction = self.clf.predict(x_test)
        if(np.any(y_test!=None)):
            self.prediction_conf_values = confusion_values(prediction, y_test)
        return(prediction)

    def predict_scene_sp(self, scene_data, r_max, c_max, array_of_patches_indices, numSegments = 200, sigma = 5, scene_name=None, patch_x_size = 384, patch_y_size = 384):
        self.scene_name=scene_name

        #print('Predicting scene')
        #margin_state = get_states(margin_point, w, V1, V2)
        for j in range(r_max):#run through all rows of the scene
            if(j%1==0): print('N:', self.N_train,'| row number: ',j)

            for i in range(c_max):#run through all columns of the scene
                #print('col number: ', i)

                patch_id = array_of_patches_indices[j,i]#identify the patch_id of the patch in current element of the scene grid
                
                #patch_p = scene_data.open_as_points(patch_id, include_nir=True)

                image = scene_data.open_as_array(idx=patch_id)
                image_nir = scene_data.open_as_array(idx=patch_id, include_nir=True)

                predicted_array = np.empty((patch_x_size, patch_y_size), dtype=int)
                segments = slic(image, n_segments = numSegments, sigma = sigma, convert2lab=True, channel_axis=2)
                for segment_id in range(1,len(np.unique(segments))+1):
                    segment_mask = (segments == segment_id)#Create segment mask
                    segment = image_nir[segment_mask]#Create segment from the patch image
                    segment_gt = np.zeros((segment.shape[0]))#Create gt segment from patch gt
                    superpixel, feature_names= summarize_segment(segment=segment, segment_gt=segment_gt) # summarize_segment should be changed to take segment_gt as an optional argument
                    superpixel = superpixel[:-1].reshape(1,-1)
                    superpixel = (superpixel-self.data_min)/(self.data_max-self.data_min)#Normalize before PCA
                    if(self.pca!=None): 
                        superpixel = self.pca.transform(superpixel)
                        superpixel = (superpixel-self.pca_min)/(self.pca_max-self.pca_min)#Normalize after PCA
                    prediction = self.clf.predict(superpixel)[0]
                    #print(prediction)
                    predicted_array[segment_mask] = prediction

                if(i==0): 
                    row = predicted_array
                else:
                    row = np.concatenate((row,predicted_array),axis=1)
            if(j==0):
                predicted_gt = row
            else:
                predicted_gt = np.concatenate((predicted_gt,row),axis=0)
        return(predicted_gt)

    def change_params(self, C):
        self.C = C
        self.clf = SVC(kernel='linear', C=C)

    def export_scene_prediction_results(self, confusion_values, dataset_name = 'SVM_XM'):
        colnames = ['N', 'tp', 'tn', 'fp', 'fn', 'SV0', 'SV1', 'C', 'timestamp', 'scene']
        result = np.array([[self.N_train, confusion_values[0], confusion_values[1], confusion_values[2], confusion_values[3], self.SV_n[0], self.SV_n[1], self.C, get_timestamp(), self.scene_name]])
        result_df = pd.DataFrame(result, columns = colnames)
        result_df.to_csv('result_'+ self.classifier_name + '_' + dataset_name +'.csv', mode = 'a', index=False, header=False)

#---------------------------------------------------------------------

def confusion_values(pred, y):
    pred = np.array(pred, dtype=bool)
    y = np.array(y, dtype=bool)

    tp = (pred*y).sum()
    tn = (np.invert(pred)*np.invert(y)).sum()
    fp = (pred*np.invert(y)).sum()
    fn = (np.invert(pred)*y).sum()

    return(tp, tn, fp, fn)

def accuracy_from_confusion_values(conf_val):
    return((conf_val[0]+conf_val[1])/(np.array(conf_val).sum()))

def get_timestamp():
    now = datetime.datetime.now()
    return((now.year-2000)*100000000 + now.month*1000000 + now.day*10000 + now.hour*100 + now.minute)


def superpixelize_patch(image_nir, image_gt, segments):
    patch_superpixels = np.empty((0,25))

    for segment_id in range(1,len(np.unique(segments))+1):
        segment_mask = (segments == segment_id)#Create segment mask

        segment_unfiltered = image_nir[segment_mask]#Create segment from the patch image
        segment_gt_unfiltered = image_gt[segment_mask]#Create gt segment from patch gt

        segment = segment_unfiltered[np.sum(segment_unfiltered, axis=1) > 0]#Filter out the margins in segment
        segment_gt = segment_gt_unfiltered[np.sum(segment_unfiltered, axis=1) > 0]#Filter out the margins in segment gt
        
        superpixel, feature_names= summarize_segment(segment=segment, segment_gt=segment_gt)#Create superpixel from the current segment
        patch_superpixels = np.vstack([patch_superpixels, superpixel])#Add superpixel to superpixels array

    return(patch_superpixels)

def summarize_segment(segment, segment_gt):

    segment_means = np.mean(segment, axis=0)
    segment_medians = np.median(segment, axis=0)
    segment_iqrs = iqr(segment, axis=0)
    segment_maxs = np.max(segment, axis = 0)
    segment_mins = np.min(segment, axis = 0)
    segment_stds = np.std(segment, axis = 0)

    segment_gt_avg = np.mean(segment_gt)

    feature_names = [
    'R_med', 'G_med', 'B_med', 'Nir_med',
    'R_avg', 'G_avg', 'B_avg', 'Nir_avg',
    'R_iqr', 'G_iqr', 'B_iqr', 'Nir_iqr',
    'R_max', 'G_max', 'B_max', 'Nir_max',
    'R_min', 'G_min', 'B_min', 'Nir_min',
    'R_std', 'G_std', 'B_std', 'Nir_std',
    'gt_avg'
    ]
    segment_features = np.array([segment_medians, segment_means, segment_iqrs, segment_maxs, segment_mins, segment_stds]).flatten()
    segment_features_gt = np.append(segment_features, segment_gt_avg)

    return(segment_features_gt, feature_names)
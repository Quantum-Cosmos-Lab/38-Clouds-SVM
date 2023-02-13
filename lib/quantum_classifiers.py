from sklearn.svm import SVC
import pennylane as qml
from pennylane import numpy as qnp
from pennylane import numpy as np
import datetime
import pandas as pd
from scipy.stats import iqr
from skimage.segmentation import slic

#####----------------------------------------------------------------------------------------

class TA_hybrid_SVM:
    def __init__(self, state_circ, circ_params, classifier_name='hybrid_TA', C=1, pca=None, data_min = 0, data_max = 1, pca_min = 0, pca_max = 1) -> None:
        self.classifier_name = classifier_name

        self.state_circ = state_circ #The quantum circuit which returns the quantum state after acting with feature map
        self.circ_params = circ_params #The parameters of quantum circuit

        self.pca = pca #Trained PCA transform
        self.data_min = data_min #array of minimal values for each feature
        self.data_max = data_max #array of maximal values for each feature
        self.pca_min = pca_min #array of minimal values for each feature after PCA
        self.pca_max = pca_max #array of maximal values for each feature after PCA

        self.clf = SVC(kernel='precomputed', C=C)
        self.C = C

        self.trained = False
        self.training_acc = None
        self.N_train = None
        self.SV_n = None
        self.SVs = None
        self.prediction_conf_values = None
        self.kernel_alignment = 0

        self.scene_name = None

    def get_states(self, X):
        states = np.array([self.state_circ(x,self.circ_params) for x in X])
        return(states)

    def train_kernel(self, states):
        k = np.absolute(np.dot(states, np.conj(states).T))**2
        return(k)

    def test_kernel(self, states):
        k = np.absolute(np.dot(states, np.conj(self.SVs).T))**2
        return(k)

    def train(self, x_train, y_train, use_SVs = True, info=True):
        """
        Method for training SVM classifier
        """
        states = self.get_states(x_train)
        kernel_train = self.train_kernel(states)
        self.clf.fit(kernel_train, y_train)

        if(self.clf.fit_status_):
            if(info):print("Not fitted")
            return(0)
        else:
            if(info):print("Fitted correctly")
            
            acc=self.clf.score(kernel_train, y_train)
            if(info):print("Fit accuracy: ", acc)
            
            self.N_train = x_train.shape[0]
            self.trained = True
            self.training_acc = acc
            self.SV_n = self.clf.n_support_
            self.kernel_alignment = self.TA(x_train, y_train)
            if(use_SVs):
                # Retrain the model, but this time use only support vectors in the training set. 
                # Takes additional time to train, but saves time in the test phase.

                states = states[self.clf.support_]
                self.SVs = states
                y_train = y_train[self.clf.support_]
                kernel_train = self.train_kernel(states)
                self.clf.fit(kernel_train, y_train)
            return(acc)

    def TA(self, X, Y, rescale_class_labels = False):
        # Calculate Kernel-Target Alignment from the data

        states = self.get_states(X)
        K = self.train_kernel(states)

        if rescale_class_labels:
            nplus = np.count_nonzero(np.array(Y) == 1)
            nminus = len(Y) - nplus
            _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
        else:
            _Y = np.array(Y)
        
        #Center class labels
        labels = _Y.copy()
        classes = np.unique(labels)
        centered_labels = (labels - classes.mean())
        _Y = centered_labels/centered_labels.max()

        T = np.outer(_Y, _Y)
        inner_product = np.sum(K * T)
        norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
        inner_product = inner_product / norm

        return inner_product

    def cost(self, theta, x, y):
        # -Target-Kernel Alignment taken as a cost function for validation phase

        self.change_params(C=1, p=theta)
        return(-self.TA(x,y))


    def random_parameters(self):
        # Create an array of random parameters for the quantum circuit
        thetas = 2 * np.pi * qnp.random.random(size=self.circ_params.shape, requires_grad=True)
        return(thetas)

    def validate(self, x_train, y_train, x_val, y_val, C_range, info=False, use_SVs=True, logscale = False, optimizer_step = 0.3, max_optimizer_steps = 50, n_initializations=10, custom_batch_size = 32):
        
        #-------------Preparation-------------

        if(logscale):
            Cs = np.logspace(C_range[0],C_range[1],C_range[2])            
        else:
            Cs = np.arange(C_range[0],C_range[1],C_range[2])

        # Initialize hyperparameters and evaluation metrics for the best model
        best_p = self.circ_params
        best_TA = 0
        best_C = 0
        best_acc = 0

        # Make sure that the optimizer doesn't calculate gradient wrt data
        x_train = qnp.array(x_train, requires_grad = False)
        y_train = qnp.array(y_train, requires_grad = False)

        # Split the X data into positive and negative class representatives
        x_train_pos = x_train[(y_train == 1)]
        x_train_neg = x_train[(y_train == 0)]

        # Sample a balanced batch from the data
        if(x_train.shape[0]//2 % 2 == 0):
            batch_size = x_train.shape[0]//2
        else:
            batch_size = x_train.shape[0]//2+1
        batch_size = min(batch_size, custom_batch_size)

        if(info): print('Batch size: ', batch_size)
        cst = [-self.TA(x_train, y_train)]

        #-------------Optimization-------------
        

        # Try small optimization from different locations in parameter space
        best_initial_TA = 0
        for _ in range(n_initializations):
            thetas = self.random_parameters()
            self.change_params(C=1, p=thetas)
            tmp_TA = self.TA(x_train, y_train)
            if(tmp_TA>best_initial_TA):
                best_initial_TA = tmp_TA
                best_p = thetas
        self.change_params(C=1, p=best_p)

        if(info):print('Optimization started!')

        # Start the optimization
        opt = qml.AdamOptimizer(optimizer_step)
        
        for step in range(max_optimizer_steps):
            # Select batch of data
            batch_index = np.random.randint(0, len(x_train_pos), (batch_size//2,))
            x_batch_pos = x_train[batch_index]
            y_batch_pos = np.ones((len(batch_index)))

            batch_index = np.random.randint(0, len(x_train_neg), (batch_size//2,))
            x_batch_neg = x_train[batch_index]
            y_batch_neg = np.zeros((len(batch_index)))

            x_batch = np.concatenate([x_batch_pos,x_batch_neg])
            y_batch = np.concatenate([y_batch_pos,y_batch_neg])

            # Update the weights by one optimizer step
            thetas = opt.step(lambda thetas: self.cost(thetas, x_batch,y_batch), thetas)

            # Save, and possibly print, the current cost
            c = self.cost(thetas, x_train, y_train)
            cst.append(c)
            if (step + 1) % 10 == 0:
                print("Cost at step {0:3}: {1}".format(step + 1, c))

        print('Optimization finished!')
        #-------------Optimization-------------
        
        print('Choosing C')
        for C in Cs:
            # Predict on validation set
            self.change_params(C=C, p=thetas)
            self.train(x_train, y_train, info=False)
            pred = self.predict(x_val)
            acc = accuracy_from_confusion_values(confusion_values(pred, y_val))

            # Change results if better
            if(acc > best_acc):
                best_acc = acc
                best_p = thetas
                best_C = C
                best_TA = -c

        if(info):print('Best validation score: ', np.round(100*best_acc, decimals=2),'%')

        if(info):print('best TA: ', best_TA)
        if(info):print('best C: ', best_C)
        self.change_params(C=best_C, p=np.array(best_p))#np.array(best_p)
        self.train(x_train,y_train, info = False)
        return(best_acc)

    def predict(self, x_test, y_test=None):
        test_states = self.get_states(x_test)
            
        kernel_test = self.test_kernel(test_states)
        prediction = self.clf.predict(kernel_test)
        if(np.any(y_test!=None)):
            self.prediction_conf_values = confusion_values(prediction, y_test)
        return(prediction)

    def predict_scene_sp(self, scene_data, r_max, c_max, array_of_patches_indices, numSegments = 200, sigma = 5, scene_name=None, patch_x_size = 384, patch_y_size = 384):
            """
            The method runs a test phase with superpixel segmentation of the scene from scene_data
            """
            
            self.scene_name=scene_name

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
                        sp_state = self.get_states(superpixel)
                        ker_test = self.test_kernel(sp_state)
                        prediction = self.clf.predict(ker_test)[0]
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

    def change_params(self, C, p):
        self.C = C
        self.circ_params = p
        self.clf = SVC(kernel='precomputed', C=C)

    def export_scene_prediction_results(self, confusion_values, dataset_name = 'SVM_XM'):
        colnames = ['N', 'tp', 'tn', 'fp', 'fn', 'SV0', 'SV1', 'C', 'p', 'timestamp', 'scene']
        result = np.array([[self.N_train, confusion_values[0], confusion_values[1], confusion_values[2], confusion_values[3], self.SV_n[0], self.SV_n[1], self.C, self.circ_params, get_timestamp(), self.scene_name]])
        result_df = pd.DataFrame(result, columns = colnames)
        result_df.to_csv('result_'+ self.classifier_name + '_' + dataset_name +'.csv', mode = 'a', index=False, header=False)

#####----------------------------------------------------------------------------------------

class hybrid_SVM:
    def __init__(self, state_circ, circ_params, classifier_name='hybrid', C=1, pca=None, data_min = 0, data_max = 1, pca_min = 0, pca_max = 1) -> None:
        self.classifier_name = classifier_name

        self.state_circ = state_circ
        self.circ_params = circ_params

        self.pca = pca
        self.data_min = data_min #array of minimal values for each feature
        self.data_max = data_max #array of maximal values for each feature
        self.pca_min = pca_min
        self.pca_max = pca_max

        self.clf = SVC(kernel='precomputed', C=C)
        self.C = C

        self.trained = False
        self.training_acc = None
        self.N_train = None
        self.SV_n = None
        self.SVs = None
        self.prediction_conf_values = None

        self.scene_name = None

    def get_states(self, X):
        states = np.array([self.state_circ(x,self.circ_params) for x in X])
        return(states)

    def train_kernel(self, states):
        k = np.absolute(np.dot(states, np.conj(states).T))**2
        return(k)

    def test_kernel(self, states):
        k = np.absolute(np.dot(states, np.conj(self.SVs).T))**2
        return(k)

    def train(self, x_train, y_train, use_SVs = True, info=True):
        
        states = self.get_states(x_train)
        kernel_train = self.train_kernel(states)
        self.clf.fit(kernel_train, y_train)

        if(self.clf.fit_status_):
            if(info):print("Not fitted")
            return(0)
        else:
            if(info):print("Fitted correctly")
            acc=self.clf.score(kernel_train, y_train)
            if(info):print("Fit accuracy: ", acc)
            self.N_train = x_train.shape[0]
            self.trained = True
            self.training_acc = acc
            self.SV_n = self.clf.n_support_
            if(use_SVs):
                states = states[self.clf.support_]
                self.SVs = states
                y_train = y_train[self.clf.support_]
                kernel_train = self.train_kernel(states)
                self.clf.fit(kernel_train, y_train)
            return(acc)

    def validate(self, x_train, y_train, x_val, y_val, C_range, p_range, info=False, use_SVs=True, logscale = False):
        ## to be extended to more parameters
        if(logscale):
            ps = np.logspace(p_range[0],p_range[1],p_range[2])
            Cs = np.logspace(C_range[0],C_range[1],C_range[2])            
        else:
            ps = np.arange(p_range[0],p_range[1],p_range[2])
            Cs = np.arange(C_range[0],C_range[1],C_range[2])

        best_p = 0
        best_C = 0
        best_acc = 0

        for C in Cs:
            for p in ps:
                self.change_params(C=C, p=p)
                self.train(x_train,y_train, info = False, use_SVs=use_SVs)
                pred = self.predict(x_val)
                acc = accuracy_from_confusion_values(confusion_values(pred, y_val))
                if(acc > best_acc):
                    best_acc = acc
                    best_p = p
                    best_C = C

        if(info):print('Best validation score: ', np.round(100*best_acc, decimals=2),'%')

        if(info):print('best p: ', best_p)
        if(info):print('best C: ', best_C)
        self.change_params(C=best_C, p=best_p)
        self.train(x_train,y_train, info = False)
        return(best_acc)

    def validate_2nd_p(self, x_train, y_train, x_val, y_val, C_range, p_range, info=False, use_SVs=True, logscale = False):
        ## to be extended to more parameters
        if(logscale):
            ps = np.logspace(p_range[0],p_range[1],p_range[2])
            Cs = np.logspace(C_range[0],C_range[1],C_range[2])            
        else:
            ps = np.arange(p_range[0],p_range[1],p_range[2])
            Cs = np.arange(C_range[0],C_range[1],C_range[2])

        best_p = 0
        best_C = 0
        best_acc = 0

        for C in Cs:
            for p in ps:
                self.change_params(C=C, p=[self.circ_params[0], p])
                self.train(x_train,y_train, info = False, use_SVs=use_SVs)
                pred = self.predict(x_val)
                acc = accuracy_from_confusion_values(confusion_values(pred, y_val))
                if(acc > best_acc):
                    best_acc = acc
                    best_p = p
                    best_C = C

        if(info):print('Best validation score: ', np.round(100*best_acc, decimals=2),'%')

        if(info):print('best p: ', best_p)
        if(info):print('best C: ', best_C)
        self.change_params(C=best_C, p=[self.circ_params[0], p])
        self.train(x_train,y_train, info = False)
        return(best_acc)

    def predict(self, x_test, y_test=None):
        test_states = self.get_states(x_test)
        kernel_test = self.test_kernel(test_states)
        prediction = self.clf.predict(kernel_test)
        if(np.any(y_test!=None)):
            self.prediction_conf_values = confusion_values(prediction, y_test)
        return(prediction)


    def predict_scene_sp(self, scene_data, r_max, c_max, array_of_patches_indices, numSegments = 200, sigma = 5, scene_name=None, patch_x_size = 384, patch_y_size = 384):
            self.scene_name=scene_name

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
                        sp_state = self.get_states(superpixel)
                        ker_test = self.test_kernel(sp_state)
                        prediction = self.clf.predict(ker_test)[0]
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

    def change_params(self, C, p):
        self.C = C
        self.circ_params = p
        self.clf = SVC(kernel='precomputed', C=C)

    def export_scene_prediction_results(self, confusion_values, dataset_name = 'SVM_XM'):
        colnames = ['N', 'tp', 'tn', 'fp', 'fn', 'SV0', 'SV1', 'C', 'p', 'timestamp', 'scene']
        result = np.array([[self.N_train, confusion_values[0], confusion_values[1], confusion_values[2], confusion_values[3], self.SV_n[0], self.SV_n[1], self.C, self.circ_params, get_timestamp(), self.scene_name]])
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



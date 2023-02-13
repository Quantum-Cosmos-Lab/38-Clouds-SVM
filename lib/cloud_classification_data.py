import numpy as np # linear algebra

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
import torch
import csv
import re
#import matplotlib.pyplot as plt
#from matplotlib.colors import LinearSegmentedColormap
#import time


class SceneDataset(Dataset):
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, scene_id, pytorch=True):
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.scene_patches = self.get_scene_patches(scene_id, r_dir)
        #self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if not f.is_dir()]
        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in self.scene_patches if not f.is_dir()]
        self.pytorch = pytorch
        
    
    def get_scene_patches(self, scene_id, r_dir):
        p = r_dir
        files = []
        for x in p.iterdir(): 
            a = re.search('.*'+str(scene_id)+'.*',str(x))
            if a is not None:
                files.append(Path(a.group()))
        return(files)
        

    def combine_files(self, r_file: Path, g_dir, b_dir,nir_dir, gt_dir):
        
        files = {'red': r_file, 
                 'green':g_dir/r_file.name.replace('red', 'green'),
                 'blue': b_dir/r_file.name.replace('red', 'blue'), 
                 'nir': nir_dir/r_file.name.replace('red', 'nir'),
                 'gt': gt_dir/r_file.name.replace('red', 'gt')}

        return files
                                       
    def __len__(self):
        
        return len(self.files)
     
    def open_as_array(self, idx, invert=False, include_nir=False):

        raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                            np.array(Image.open(self.files[idx]['green'])),
                            np.array(Image.open(self.files[idx]['blue'])),
                           ], axis=2)
    
        if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)
    
        if invert:
            raw_rgb = raw_rgb.transpose((2,0,1))
    
        # normalize
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)
    
    def open_as_points(self, idx, include_nir=False, reduce_margins=False):
        points = self.open_as_array(idx, include_nir=include_nir)
        points = points.reshape(points.shape[0]*points.shape[1], points.shape[2])
        
        if(reduce_margins):
            patch_mask = np.logical_or.reduce(points, 1).astype(bool)
            points = points[patch_mask]
        
        return(points)

    def open_mask(self, idx, add_dims=False):
        
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask==255, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def open_mask_as_points(self, idx, include_nir=False, reduce_margins=False):
        points = self.open_mask(idx)
        points = points.reshape(points.shape[0]*points.shape[1])

        if(reduce_margins):
            mask = self.open_as_array(idx, include_nir=include_nir)
            mask = mask.reshape(mask.shape[0]*mask.shape[1], mask.shape[2])
            mask = np.logical_or.reduce(mask, 1).astype(bool)
            points = points[mask]

        return(points)

    def points_to_mask(self, idx, points):
        mask = self.open_mask(idx)
        return(points.reshape(mask.shape[0], mask.shape[1]))

    def __getitem__(self, idx):
        
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch, include_nir=True), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)
        
        return x, y
    
    def open_as_pil(self, idx):
        
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())

        return s

def import_scene_ids(scene_id_path):
    with open(scene_id_path, newline='') as f:
        reader = csv.reader(f)
        scene_id_unflattened = list(reader)[1:]
    
    scene_id = []
    for item in scene_id_unflattened:
        scene_id.append(item[0])
    return(scene_id)

def import_scene_names(scene_id_path):
    scene_id_path = Path(scene_id_path)
    with open(scene_id_path, newline='') as f:
        reader = csv.reader(f)
        scene_id_unflattened = list(reader)[1:]
    
    scene_id = []
    for item in scene_id_unflattened:
        scene_id.append(item[0])
    return(scene_id)

def import_scene(scene_id, scene_names, base_path):
    scene = scene_names[scene_id]
    scene_data = SceneDataset(base_path/'test_red',
    base_path/'test_green',
    base_path/'test_blue',
    base_path/'test_nir',
    base_path/'test_gt', scene)
    return(scene_data)

def get_scene(base_path, scene_id):
    base_path = Path(base_path)
    scenes_path = base_path/'test_sceneids_38-Cloud.csv'
    scene_names = import_scene_names(scenes_path)
    scene_data = import_scene(scene_id=scene_id, scene_names=scene_names, base_path=base_path)
    patches_paths = scene_data.get_scene_patches(scene_names[scene_id], base_path/'test_red')
    r_max, c_max = get_scene_array_size(patches_paths)
    patches_ids_array = get_array_of_patches_ids(r_max, c_max, patches_paths)
    return(scene_data, r_max, c_max, patches_ids_array, scene_names)

def get_row_col(patch_id):
    patch_id = str(patch_id)
    start = re.search('patch', patch_id)
    end = re.search('LC08', patch_id)
    patch_name = patch_id[start.span()[1]:end.span()[0]]
    indices = [i.start() for i in re.finditer('_', patch_name)]
    row = int(patch_name[(indices[1]+1):indices[2]])
    col = int(patch_name[(indices[3]+1):indices[4]])
    return(row, col)

def get_scene_array_size(patches_path):
    r_max, c_max = 0,0
    for patch_id in patches_path:
        r, c = get_row_col(patch_id=patch_id)
        if(r > r_max): r_max = r
        if(c > c_max): c_max = c
    return(r_max, c_max)

def get_array_of_patches_ids(r_max, c_max, patches_path):
    array_of_patches_indices = np.zeros((r_max,c_max), dtype=int)
    for i in range(len(patches_path)):
        r, c = get_row_col(patch_id=patches_path[i])
        array_of_patches_indices[r-1,c-1] = i
    return(array_of_patches_indices)


def get_scene_mask(gt_path, scene_names, scene_id):
    gt_filename = 'edited_corrected_gts_'+scene_names[scene_id] + '.TIF'
    gt_scene_image = Image.open(gt_path/gt_filename)
    gt_scene = np.array(gt_scene_image)
    return(gt_scene)

def unzeropad(in_dest, in_source):
    ny, nx = in_dest.shape
    nys, nxs = in_source.shape
    #print(ny, nx)

    tmpy = int(np.floor((ny-nys)/2.0))
    tmpx = int(np.floor((nx-nxs)/2.0))
    #print(tmpy)

    return(in_dest[(tmpy):(tmpy+nys), (tmpx):(tmpx+nxs)])
from config import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import math
import pickle
import os
from src.dataset.uv_face import mean_shape_map_np, face_mask_np, foreface_ind, uv_coords, uv_kpt_ind
from io import BytesIO


class ImageData:
    def __init__(self):
        self.image_path = ""
        self.uv_posmap_path = ""

        self.pos_map = None
        self.image = None

    def read_path(self, image_dir):
        image_name = image_dir.split('/')[-1]
        self.image_path = image_dir + '/' + image_name + '.jpg'
        self.uv_posmap_path = image_dir + '/' + image_name + '_pos_map.npy'

    def get_image(self):
        if self.image is None:
            return np.array(Image.open(self.image_path).convert('RGB'))
        else:
            return np.array(Image.open(BytesIO(self.image)).convert('RGB'))

    def get_pos_map(self):
        if self.pos_map is None:
            pos_map = np.load(self.uv_posmap_path).astype(np.float32)
        else:
            pos_map = self.pos_map.astype(np.float32)

        return pos_map


class FaceRawDataset:
    def __init__(self):
        self.train_data = []
        self.val_data = []
        self.test_data = []

    def add_image_data(self, data_dir, add_mode='train', split_rate=0.8, save_posmap_num=0):
        all_data = []
        saved_num = 0
        if os.path.exists(f'{data_dir}/all_image_data.pkl'):
            all_data = self.load_image_data_paths(data_dir)
        else:
            for root, dirs, files in os.walk(data_dir):
                dirs.sort()
                for dir_name in dirs:
                    image_name = dir_name
                    if not os.path.exists(root + '/' + dir_name + '/' + image_name + '.jpg'):
                        print('skip ', root + '/' + dir_name)
                        continue
                    temp_image_data = ImageData()
                    temp_image_data.read_path(root + '/' + dir_name)
                    if saved_num < save_posmap_num:
                        saved_num += 1
                        temp_image_data.pos_map = np.load(temp_image_data.uv_posmap_path)

                    all_data.append(temp_image_data)
                    print(f'\r{len(all_data)}', end='')

        print(len(all_data), 'data added')

        if add_mode == 'train':
            self.train_data.extend(all_data)
        elif add_mode == 'val':
            self.val_data.extend(all_data)
        elif add_mode == 'both':
            num_train = math.floor(len(all_data) * split_rate)
            self.train_data.extend(all_data[0:num_train])
            self.val_data.extend(all_data[num_train:])
        elif add_mode == 'test':
            self.test_data.extend(all_data)

    def load_image_data_paths(self, data_dir):
        print('loading data path list')
        ft = open(f'{data_dir}/all_image_data.pkl', 'rb')
        all_data = pickle.load(ft)
        ft.close()
        print('data path list loaded')
        return all_data


def img_to_tensor(image):
    return torch.from_numpy(image.transpose((2, 0, 1)))


def make_dataset(folders, mode='train'):
    raw_dataset = FaceRawDataset()
    for folder in folders:
        raw_dataset.add_image_data(folder, mode)
    return raw_dataset

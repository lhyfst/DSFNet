from src.configs.config_DSFNet_eval import *
    
INIT_IMAEG_SIZE = 450
CROPPED_IMAGE_SIZE = 256
UV_MAP_SIZE = 256

UV_MEAN_SHAPE_PATH = 'data/uv_data/mean_shape_map.npy'
UV_FACE_MASK_PATH = 'data/uv_data/uv_face_mask.png'
BFM_UV_MAT_PATH = 'data/Out/BFM_UV.mat'
UV_KPT_INDEX_PATH = 'data/uv_data/uv_kpt_ind.txt'
DEVICE = 'cuda:0'
import src.faceutil
from src.faceutil.morphable_model import MorphabelModel
from PIL import Image
from config import *
import trimesh
import os
import numpy as np

face_mask_np = np.array(Image.open(UV_FACE_MASK_PATH)) / 255.
face_mask_fix_rate = (UV_MAP_SIZE ** 2) / np.sum(face_mask_np)
foreface_ind = np.array(np.where(face_mask_np > 0)).T

if os.path.exists(UV_MEAN_SHAPE_PATH):
    mean_shape_map_np = np.load(UV_MEAN_SHAPE_PATH)

def process_uv(uv_coordinates):
    uv_h = UV_MAP_SIZE
    uv_w = UV_MAP_SIZE
    uv_coordinates[:, 0] = uv_coordinates[:, 0] * (uv_w - 1)
    uv_coordinates[:, 1] = uv_coordinates[:, 1] * (uv_h - 1)
    uv_coordinates[:, 1] = uv_h - uv_coordinates[:, 1] - 1
    uv_coordinates = np.hstack((uv_coordinates, np.zeros((uv_coordinates.shape[0], 1))))  # add z
    return uv_coordinates

def read_uv_kpt(uv_kpt_path):
    file = open(uv_kpt_path, 'r', encoding='utf-8')
    lines = file.readlines()
    # txt is inversed
    x_line = lines[1]
    y_line = lines[0]
    uv_kpt = np.zeros((68, 2)).astype(int)
    x_tokens = x_line.strip().split(' ')
    y_tokens = y_line.strip().split(' ')
    for i in range(68):
        uv_kpt[i][0] = int(float(x_tokens[i]))
        uv_kpt[i][1] = int(float(y_tokens[i]))
    return uv_kpt


uv_coords = src.faceutil.morphable_model.load.load_uv_coords(BFM_UV_MAT_PATH)
uv_coords = process_uv(uv_coords)
uv_kpt_ind = read_uv_kpt(UV_KPT_INDEX_PATH)


def get_kpt_from_uvm(uv_map):
    # from uv map
    kpt = uv_map[uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]]
    return kpt


def uvm2mesh(uv_position_map, only_foreface=True, is_extra_triangle=False):
    """
    if no texture map is provided, translate the position map to a point cloud
    :param uv_position_map:
    :param uv_texture_map:
    :param only_foreface:
    :return:
    """
    uv_h, uv_w = UV_MAP_SIZE, UV_MAP_SIZE
    vertices = []
    colors = []
    triangles = []
    triangles = [[t[0] * uv_w + t[1], t[4] * uv_w + t[5], t[2] * uv_w + t[3], ] for t in uv_triangles]
    for i in range(uv_h):
        for j in range(uv_w):
            vertices.append(uv_position_map[i][j])
            colors.append([25, 25, 50, 128])

    vertices = np.array(vertices)
    triangles = np.array(triangles)
    colors = np.array(colors)
    face_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    return face_mesh


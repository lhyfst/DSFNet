import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import config
from src.dataset.dataloader import make_dataset, img_to_tensor
from src.model.loss import *
from src.util.printer import DecayVarPrinter
from tqdm import tqdm
import skimage.io as io
from src.faceutil import mesh

class BasePredictor:
    def __init__(self, weight_path):
        self.model = self.get_model(weight_path)

    def get_model(self, weight_path):
        raise NotImplementedError

    def predict(self, img):
        raise NotImplementedError


class DSFNetPredictor(BasePredictor):
    def __init__(self, weight_path):
        super(DSFNetPredictor, self).__init__(weight_path)

    def get_model(self, weight_path):
        from src.model.DSFNet import get_model
        model = get_model()
        pretrained = torch.load(weight_path, map_location=config.DEVICE)
        model.load_state_dict(pretrained)
        model = model.to(config.DEVICE)
        model.eval()
        return model


class Evaluator:
    def __init__(self):
        self.all_eval_data = None
        self.metrics = {"nme3d": NME(),
                        "nme2d": NME2D(),
                        "kpt2d": KptNME2D(),
                        "kpt3d": KptNME(),
                        "rec": RecLoss(),
                        }
        self.printer = DecayVarPrinter()

    def get_data(self):
        val_dataset = make_dataset(config.VAL_DIR, 'val')
        self.all_eval_data = val_dataset.val_data

    def evaluate(self, predictor):
        with torch.no_grad():
            predictor.model.eval()
            self.printer.clear()

            pred_angles = np.zeros((len(self.all_eval_data),3))
            valid_idx = np.arange(len(self.all_eval_data))
            for i in tqdm(valid_idx):
                item = self.all_eval_data[i]
                init_img = item.get_image()
                image = (init_img / 255.0).astype(np.float32)
                for ii in range(3):
                    image[:, :, ii] = (image[:, :, ii] - image[:, :, ii].mean()) / np.sqrt(
                        image[:, :, ii].var() + 0.001)
                image = img_to_tensor(image).to(config.DEVICE).float().unsqueeze(0)
        
                init_pos_map = (item.get_pos_map())
                pos_map = init_pos_map / config.POSMAP_FIX_RATE
                pos_map = img_to_tensor(pos_map).float().to(config.DEVICE).unsqueeze(0)

                out = predictor.model(inpt={'img': image}, targets={}, mode='predict')
                pred_a = mesh.transform.matrix2angle(out['R_rot'].cpu().detach().numpy()[0].T)
                pred_a = np.array([pred_a[0],pred_a[1],pred_a[2]])
                pred_a[0] = -pred_a[0]
                pred_angles[i] = pred_a
    
                for key in self.metrics:
                    func = self.metrics[key]
                    error = func(pos_map, out['face_uvm']).cpu().numpy()
                    self.printer.update_variable_avg(key, error)


        print('Dataset Results')
        return_dict = {}
        for key in self.metrics:
            print(self.printer.get_variable_str(key))
            return_dict[key] = float(self.printer.get_variable_str(key).split(' ')[1])
            
        head_pose_estimation = benchmark_FOE(pred_angles,valid_idx)
        for key in head_pose_estimation:
            return_dict[key] = round(head_pose_estimation[key],3)
    
        return return_dict
    

if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.get_data()
    predictor = DSFNetPredictor(config.PRETAINED_MODEL)
    evaluator.evaluate(predictor)

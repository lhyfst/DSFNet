# DSFNet

Paper link: [https://arxiv.org/abs/2305.11522](https://arxiv.org/abs/2305.11522)
Project link: [https://lhyfst.github.io/dsfnet/](https://lhyfst.github.io/dsfnet/)
Video link: [https://www.youtube.com/watch?v=tNcI-1Y9FW8](https://www.youtube.com/watch?v=tNcI-1Y9FW8)

## Requirements
```
python                    3.6.13
pytorch                   1.7.1
cudatoolkit               10.1.243
imageio                   2.15.0
numpy                     1.19.2
opencv-python             4.7.0.72
PyYAML                    6.0
scikit-image              0.17.2
torchvision               0.8.2
tqdm                      4.64.1
trimesh                   3.22.1
```


## Prepare

* Please refer to [face3d](https://github.com/YadiraF/face3d/blob/master/examples/Data/BFM/readme.md) to prepare BFM data. And move the generated files in ```Out/``` to ```data/Out/``` 

* Download [BFM_UVspace_patch.npy](https://drive.google.com/file/d/15q5I7bgZQOWGxXnNWt0Drg__SZ0CsoWJ/view?usp=sharing). Put it under ```data/uv_data/```

* Download [pretrained model](https://drive.google.com/file/d/1YdzmY7i1pN_mmkPAZsLmA7yp2-TCwR7x/view?usp=drive_link). Put it under ```data/saved_model/```.



## Evaluation

* Download AFLW2000-3D at [http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3ddfa/main.htm](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3ddfa/main.htm) .

* Follow [SADRNet](https://github.com/MCG-NJU/SADRNet) to crop images and prepare the image directory. Or you can download the cropped images at [link](https://drive.google.com/file/d/1NX1uN8o6vVYw2z4JL7gxZqFBYXnmeYyR/view?usp=sharing). Put them at ```data/dataset/AFLW2000_crop```.

* Run ```src/run/predict.py```. In the returned text, nme3d, rec, MAE are the results of dense 3D dense face alignment, reconstruction, and head pose estimation. 


## Acknowledgements
We especially thank the contributors of the [SADRNet](https://github.com/MCG-NJU/SADRNet) codebase for providing helpful code.
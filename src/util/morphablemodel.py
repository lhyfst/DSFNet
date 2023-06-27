import numpy as np
import torch
import sys
import os
try:
    from . import load
except:
    import load


class MorphableModel:
    def __init__(self, model_path, model_type='BFM_UV', device='cuda'):
        '''
        a statistical morphable model is a generative model that can generate faces with different identity, expression and skin reflectance
        it is mainly composed of an orthogonal basis (eigen vectors) obtained from applying principal component analysis (PCA) on a set of face scans.
        a linear combination of these eigen vectors produces different type shape and skin
        :param path: drive path of where the data of the morphable model is saved
        :param textureResolution: the resolution of the texture used for diffuse and specular reflectance
        :param trimPca: if True keep only a subset of the PCA basis
        :param landmarksPathName: a text file conains the association between the 2d pixel position and the 3D points in the mesh
        :param device: where to store the morphableModel data (cpu or gpu)
        '''
        self.shapeBasisSize = 199
        self.expBasisSize = 29
        self.device = device

        if model_type=='BFM':
            self.model = load.load_BFM(model_path)
        elif model_type=='BFM_UV':
            self.model = load.load_BFM_UVspace(model_path)
            
        else:
            print('sorry, not support other 3DMM model now')
            exit()        
            
        self.nver = int(self.model['shapePC'].shape[0]/3)
        self.ntri = self.model['tri'].shape[0]
        self.n_shape_para = self.model['shapePC'].shape[1]
        self.n_exp_para = self.model['expPC'].shape[1]
        
        self.shapeBasisSize = 199
        self.expBasisSize = 29
        self.device = device
        
        self.shapeMean = torch.tensor(self.model['shapeMU'].reshape(self.nver,3)).to(device) # (65536, 3)
        self.shapePca = torch.tensor(self.model['shapePC'].T.reshape(self.shapeBasisSize, self.nver, 3)).to(device) # (199, 65536, 3)
        self.shapePcaVar = torch.tensor(self.model['shapeEV'].reshape(self.shapeBasisSize)).to(device) # (199)

        self.expressionPca = torch.tensor(self.model['expPC'].T.reshape(self.expBasisSize,self.nver,3)).to(device) # (29, 65536, 3)
        self.expressionPcaVar = torch.tensor(self.model['expEV'].reshape(self.expBasisSize)).to(device) # (29)
        
        self.kpt_ind = torch.tensor(self.model['kpt_ind'].reshape(-1)).long().to(device) # (68)
        
    
    def computeShape(self, shapeCoff, expCoff):
        '''
        compute vertices from shape and exp coeff
        :param shapeCoff: [n, self.shapeBasisSize]
        :param expCoff: [n, self.expBasisSize]
        :return: return vertices tensor [n, verticesNumber, 3]
        '''
        assert (shapeCoff.dim() == 2 and shapeCoff.shape[1] == self.shapeBasisSize)
        assert (expCoff.dim() == 2 and expCoff.shape[1] == self.expBasisSize)
        shapeCoff = shapeCoff * self.shapePcaVar[None,...]
        expCoff = expCoff * self.expressionPcaVar[None,...]

        vertices = self.shapeMean + torch.einsum('ni,ijk->njk', (shapeCoff, self.shapePca)) + torch.einsum('ni,ijk->njk', (expCoff, self.expressionPca))
        return vertices
    
    def compute_offset_uvm(self, shapeCoff, expCoff):
        '''
        compute vertices from shape and exp coeff
        :param shapeCoff: [n, self.shapeBasisSize]
        :param expCoff: [n, self.expBasisSize]
        :return: return vertices tensor [n, verticesNumber, 3]
        '''
        assert (shapeCoff.dim() == 2 and shapeCoff.shape[1] == self.shapeBasisSize)
        assert (expCoff.dim() == 2 and expCoff.shape[1] == self.expBasisSize)
        bs = shapeCoff.shape[0]
        shapeCoff = shapeCoff * self.shapePcaVar[None,...]
        expCoff = expCoff * self.expressionPcaVar[None,...]

        vertices = torch.einsum('ni,ijk->njk', (shapeCoff, self.shapePca)) + torch.einsum('ni,ijk->njk', (expCoff, self.expressionPca))
        offset_uvm = vertices.reshape(bs, 256, 256, 3).permute(0,3,1,2) * 1.5e-5
        return offset_uvm


    def sample(self, shapeNumber = 1):
        '''
        random sample shape, expression, diffuse and specular albedo coeffs
        :param shapeNumber: number of shapes to sample
        :return: shapeCoeff [n, self.shapeBasisSize], expCoeff [n, self.expBasisSize], diffCoeff [n, albedoBasisSize], specCoeff [n, self.albedoBasisSize]
        '''
        shapeCoeff = self.sampler.sample(shapeNumber, self.shapePcaVar)
        expCoeff = self.sampler.sample(shapeNumber, self.expressionPcaVar)
        diffAlbedoCoeff = self.sampler.sample(shapeNumber, self.diffuseAlbedoPcaVar)
        specAlbedoCoeff = self.sampler.sample(shapeNumber, self.specularAlbedoPcaVar)
        return shapeCoeff, expCoeff, diffAlbedoCoeff, specAlbedoCoeff



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


### initalize the module
def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size+(n_concat-2)*out_size, out_size, False)
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
           
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)


class UNet(nn.Module):

    def __init__(self, in_channels=6, n_classes=3, feature_scale=2):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels, filters[0], True)
        self.conv2 = unetConv2(filters[0], filters[1], True)
        self.conv3 = unetConv2(filters[1], filters[2], True)
        self.conv4 = unetConv2(filters[2], filters[3], True)
        self.center = unetConv2(filters[3], filters[4], True)


        self.fuse_with_feature = unetConv2(filters[4]+72, filters[4], True)
            

        self.up_concat4_offset = unetUp(filters[4], filters[3])
        self.up_concat3_offset = unetUp(filters[3], filters[2])
        self.up_concat2_offset = unetUp(filters[2], filters[1])
        self.up_concat1_offset = unetUp(filters[1], filters[0])
        
        self.up_concat4_kpt = unetUp(filters[4], filters[3])
        self.up_concat3_kpt = unetUp(filters[3], filters[2])
        self.up_concat2_kpt = unetUp(filters[2], filters[1])
        self.up_concat1_kpt = unetUp(filters[1], filters[0])


        self.offset_head = nn.Sequential(nn.Conv2d(filters[0], filters[0], 3, 1, 1),
                             nn.BatchNorm2d(filters[0]),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(filters[0], 3, 1))
    
        self.kpt_head = nn.Sequential(nn.Conv2d(filters[0], filters[0], 3, 1, 1),
                             nn.BatchNorm2d(filters[0]),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(filters[0], 3, 1))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
                
    def forward(self, inputs, feature=None):          
        conv1 = self.conv1(inputs)     
        maxpool1 = self.maxpool(conv1)  
        
        conv2 = self.conv2(maxpool1)   
        maxpool2 = self.maxpool(conv2)  

        conv3 = self.conv3(maxpool2)    
        maxpool3 = self.maxpool(conv3)   

        conv4 = self.conv4(maxpool3)    
        maxpool4 = self.maxpool(conv4)  

        center = self.center(maxpool4)      
        feature = F.interpolate(feature, size=(center.shape[2], center.shape[3]), mode='bilinear', align_corners=False)
        center = self.fuse_with_feature(torch.cat([center, feature],dim=1))
        
        up4_offset = self.up_concat4_offset(center,conv4) 
        up3_offset = self.up_concat3_offset(up4_offset,conv3)  
        up2_offset = self.up_concat2_offset(up3_offset,conv2)  
        up1_offset = self.up_concat1_offset(up2_offset,conv1)   
        offset = self.offset_head(up1_offset)          
        
        up4_kpt = self.up_concat4_kpt(center,conv4) 
        up3_kpt = self.up_concat3_kpt(up4_kpt,conv3)  
        up2_kpt = self.up_concat2_kpt(up3_kpt,conv2)  
        up1_kpt = self.up_concat1_kpt(up2_kpt,conv1)  
        kpt = self.kpt_head(up1_kpt)  
        
        return offset, kpt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import torchvision.models as models

class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, norm='instance_norm', act='lrelu'):
        super(ConvLayer, self).__init__()
        
        self.convs = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        
        if act == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        else:
            self.act = nn.Identity()            
            
        if norm == 'instance_norm': 
            self.norm = nn.InstanceNorm2d(out_ch)
        else:
            self.norm = nn.Identity()
        
    def forward(self, x):
        x = self.convs(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    

class FC(nn.Module):
    def __init__(self, in_ch, out_ch, act='None', flatten = 'flatten'):
        super(FC, self).__init__()
        if flatten == 'flatten':
            self.flatten = nn.Flatten()
        else:
            self.flatten = nn.Identity()
        self.fc = nn.Linear(in_ch, out_ch)
        if act == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        else:
            self.act = nn.Identity()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = self.act(x)
        return x
    

class HRImageEncoder(nn.Module):
    def __init__(self, in_res=256, in_ch=3, depth=6, kernel_size=3, out_ch=64):
        super(HRImageEncoder, self).__init__()
        
        convs = []
        out_chs, out_res = [], [in_res]
        for i in range(depth):
            if i == 0:
                out_chs.append(out_ch)
                out_res.append(in_res)
                convs.append(ConvLayer(in_ch, out_ch, kernel_size=kernel_size, norm=None))
            else:
                out_chs.append(out_ch)
                out_res.append(in_res)
                convs.append(ConvLayer(prev_ch, out_ch,kernel_size=kernel_size))

            prev_ch = out_ch
            
            out_ch *= 2
            if out_ch > 1024:
                out_ch = 1024
            
            if i != (depth - 1):
                in_res //= 2
                out_chs.append(out_ch)
                out_res.append(in_res)
                convs.append(ConvLayer(prev_ch, out_ch, kernel_size=kernel_size, stride=2))    
                prev_ch = out_ch 
            
            
                
        self.convs = nn.ModuleList(convs)
        self.fc = FC(1024*out_res[-1]**2, 512)

        
    def forward(self, x):
        for i, layer in enumerate(self.convs):
            x = layer(x)
            
        x = self.fc(x) # [B, 6144]
        return x.view(-1, 512)
    
    
class SpectrogramEncoder(nn.Module):
    def __init__(self, in_res=256, in_ch=1, depth=6, kernel_size=3, out_ch=64, output_feature_size=512, max_filters=2048):
        super(SpectrogramEncoder,self).__init__()
            
        norm_fn = None
        
        convs = []
        out_chs, out_res = [], [in_res]
        in_res //=2
        out_chs.append(out_ch)
        out_res.append(in_res)
        convs.append(ConvLayer(in_ch, out_ch, kernel_size=kernel_size, stride=2, norm=norm_fn))
        prev_ch = out_ch
        
        for i in range(depth):
            out_chs.append(out_ch)
            out_res.append(in_res)
            convs.append(ConvLayer(prev_ch, out_ch, kernel_size=kernel_size, stride=1, norm=norm_fn))
            prev_ch = out_ch
            
            out_ch *= 2
            
            if out_ch > max_filters:
                out_ch = max_filters
            
            in_res //=2    
            out_chs.append(out_ch)
            out_res.append(in_res)    
            convs.append(ConvLayer(prev_ch, out_ch, kernel_size=kernel_size, stride=2, norm=norm_fn))
            prev_ch = out_ch
            
        self.convs = nn.ModuleList(convs)
        self.fc = FC(2048*2*2, 8192, act='lrelu')
        self.fc_nonF = FC(8192, output_feature_size, act=None, flatten=None)
        self.output_feature_size = output_feature_size
        
    def forward(self, x):
        for i, layer in enumerate(self.convs):
            x = layer(x)
        x = self.fc(x)
        x = self.fc_nonF(x)
        return x.view(-1, self.output_feature_size)
            

if __name__ == '__main__':
    image = torch.randn((20,1,256,256))
    model = SpectrogramEncoder()
    latent = model(image)
    print(latent.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from ..builder import ATTENTIONS

import numpy as np
import pickle
from pytorch_msssim import SSIM

class Flatten(BaseModule):
    def forward(self, x):
        return x.view(x.size(0), -1)

class permute_ln(nn.Module):
    def __init__(self, direction: bool):
        super().__init__()
        self.direction = direction
    def forward(self, x):
        if self.direction: x = x.permute(0,2,3,1)
        else: x = x.permute(0,3,1,2)
        return x

class ChannelGateLN(BaseModule):
    def __init__(
                self, 
                gate_channel, 
                reduction_ratio=16, 
                num_layers=1
                ):
        super(ChannelGateLN, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() ) # (B,C,H,W) -> (B,C*H*W)
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_norm_%d'%(i+1), nn.LayerNorm(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d(in_tensor,(in_tensor.size(2),in_tensor.size(3)), 
                            stride=(in_tensor.size(2), in_tensor.size(3)))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class SpatialGateLN(BaseModule):
    def __init__(
                self, 
                gate_channel, 
                reduction_ratio=16, 
                dilation_conv_num=2, 
                dilation_val=4
                ):
        super(SpatialGateLN, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_ln_head_reduce0', permute_ln(True))
        self.gate_s.add_module('gate_s_norm_reduce0', nn.LayerNorm(gate_channel//reduction_ratio) )
        self.gate_s.add_module('gate_s_ln_tail_reduce0', permute_ln(False))
        self.gate_s.add_module('gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module('gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module('gate_s_ln_head_di_%d'%i, permute_ln(True))
            self.gate_s.add_module('gate_s_norm_di_%d'%i, nn.LayerNorm(gate_channel//reduction_ratio) )
            self.gate_s.add_module('gate_s_ln_tail_di_%d'%i, permute_ln(False))
            self.gate_s.add_module('gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)

@ATTENTIONS.register_module()
class BAM_LN(BaseModule):
    def __init__(self, in_channels):
        super(BAM_LN, self).__init__()
        self.channel_att = ChannelGateLN(in_channels)
        self.spatial_att = SpatialGateLN(in_channels)

    def forward(self,in_tensor):
        #@ VISUALIZATION for best model
        pkl = './exp/nus/viz/'
        # if not os.path.exists(pkl): os.makedirs(pkl)
        # f_cam, f_map = in_tensor.chunk(2, dim=1)
        # f_cam, f_map = f_cam.squeeze(), f_map.squeeze()
        
        # fig, bar = {}, {}
        # fig['c'] = f_cam.sum(0).cpu().numpy()
        # fig['m'] = f_map.sum(0).cpu().numpy()
        # bar['c'] = f_cam.sum((1,2)).cpu().numpy()
        # bar['m'] = f_map.sum((1,2)).cpu().numpy()

        chn_att = self.channel_att(in_tensor)
        spa_att = self.spatial_att(in_tensor)
        # fig['a'] = spa_att[0,0,:,:].cpu().numpy()
        # bar['a'] = chn_att[0,:,0,0].cpu().numpy()

        # with open(pkl+'fig.pkl', 'ab+') as figs:
        #     pickle.dump(fig, figs)
        # with open(pkl+'bar.pkl', 'ab+') as bars:
        #     pickle.dump(bar, bars)

        # maps too large to store, store stats instead
        # infos to get map stored in map.pkl
        ssim = SSIM(data_range=1, size_average=True, channel=1, nonnegative_ssim=True)
        maps = open(pkl+'curT.pkl', 'rb')
        hdmap = pickle.load(maps)
        maps.close()
        sim = []
        # stored hdmaps in non-encoded resolution
        hdmap = torch.nn.functional.interpolate(hdmap, 
            size=spa_att.size()[-2:], mode='bilinear')
        spatial = torch.sigmoid(spa_att[:,0:1]) # keep the indexed dim
        for i in range(hdmap.size(1)):
            sim.append(ssim(hdmap[:,i:i+1], spatial))
        sims = open(pkl+'sim.pkl', 'ab')
        pickle.dump(np.asarray(sim), sims)
        sims.close()

        att = 1 + torch.sigmoid(spa_att * chn_att)
        return att * in_tensor

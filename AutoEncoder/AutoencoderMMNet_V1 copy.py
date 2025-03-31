#!/usr/bin/env python

"""
Created on Fri Dec 7 14:42:00 2021
@author: tirtha
Original Data:
x_ini.shape : torch.Size([16, 3]), x_pos.shape : torch.Size([16384, 3]), data.x_batch.shape : torch.Size([16384])
pred.shape : torch.Size([16, 2048, 3]), gt_pcd.shape : torch.Size([32768, 3])

my data: ->np.random.choice(gt_cloud.shape[0],102400,replace=True)
x_ini.shape : torch.Size([16, 3]), x_pos.shape : torch.Size([16384, 3]), data.x_batch.shape : torch.Size([16384])
pred.shape : torch.Size([16, 2048, 3]), gt_pcd.shape : torch.Size([1638400, 3])

my data: ->np.random.choice(gt_cloud.shape[0],307200,replace=True)
x_ini.shape : torch.Size([16, 3]), x_pos.shape : torch.Size([16384, 3]), data.x_batch.shape : torch.Size([16384])
pred.shape : torch.Size([16, 2048, 3]), gt_pcd.shape : torch.Size([4915200, 3])

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Sequential as Seq, ReLU, GELU
from torch.nn import Dropout, Softmax, Linear, LayerNorm
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, DynamicEdgeConv
from torch_geometric.nn.pool.topk_pool import topk
#from torch_geometric.nn.models import MLP
from torch.autograd import Variable
# from torchsummary import summary
from torchinfo import summary
from contextlib import redirect_stdout



# ##### MODELS: Generator model and discriminator model
def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Linear(channels[i - 1], channels[i]), ReLU())
        #               ,BN(channels[i]))
        for i in range(1, len(channels))
    ])
    
class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class TopK_PointNetPP(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(TopK_PointNetPP, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn,add_self_loops = True)

    def forward(self, x, score, pos, batch, num_samples=16):
        #find top k score
        idx = topk(score, self.ratio, batch, min_score=None)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=num_samples)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch
   
class PointNetPP(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(PointNetPP, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn,add_self_loops = True)

    def forward(self, x, pos, batch, num_samples=16):

        idx = fps(pos, batch, self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=num_samples)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class DGCNN(torch.nn.Module):
    def __init__(self,nn, k=16, aggr='max'):
        super(DGCNN, self).__init__()
        self.conv = DynamicEdgeConv(nn,k,aggr)
        self.activ = ReLU()
    def forward(self, x, pos, batch):
        # print(f"Inside DGCNN: Before conv: x.shape: {x.shape}, batch.shape: {batch.shape}")
        x = self.conv(x, batch)
        # print(f"Inside DGCNN: x.shape: {x.shape}, batch.shape: {batch.shape}")
        x = self.activ(x)
        # print(f"Inside DGCNN: After activation x.shape: {x.shape}, batch.shape: {batch.shape}")

        return x
    
class GlobalPool(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalPool, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)

        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

    
class MMGraphExtractor(torch.nn.Module):
    '''
    shape code extractor based on graph neural network
    '''
    def __init__(self):
        super(MMGraphExtractor,self).__init__()
        # pointNet++ module
        #self.top_module = TopK_PointNetPP(0.7, 0.5, MLP([576 + 3,256, 128]))
        self.pn1_module = PointNetPP(0.25, 1, MLP([1 + 3,64, 256]))
        self.pn2_module = PointNetPP(0.25, 2, MLP([256+3,256, 384]))
        #self.pn3_module = PointNetPP(0.25, 1, MLP([256 + 3, 384]))
        self.gn1_module = GlobalPool(MLP([384+3, 384, 512]))
        
    def forward(self, fea, pos, score, pi):
        '''

        '''
        # fea, pos, batch
        #top_out = self.top_module(fea, score, pos, pi)
        score = score.view(-1,1)
        pn1_out = self.pn1_module(score, pos, pi)
        #pn1_out = self.pn1_module(*top_out)
        pn2_out = self.pn2_module(*pn1_out)
        #pn3_out = self.pn3_module(*pn2_out)
        gn1_out = self.gn1_module(*pn2_out)
        #pn3_out = self.pn3_module(*pn2_out)
        return gn1_out[0]
 
class NoiseScoreNet(torch.nn.Module):
    '''
    vision extractor with for 2D images
    '''
    def __init__(self):
        super(NoiseScoreNet,self).__init__()
        self.dg1 = DGCNN(MLP([3*2, 32, 64]))
        self.dg2 = DGCNN(MLP([64*2, 128,256]))
        #self.dg3 = DGCNN(MLP([128*2,256,384]))
        #self.mlp = MLP([256,256,384])
        self.head = nn.Sequential(
                nn.Linear(576,256),
                nn.ReLU(),
                nn.Linear(256,64),
                nn.Linear(64,1)
        )
        self.activ = nn.Sigmoid()

    def forward(self, x, pos, batch):
        #print(x.size())
        batch_num = torch.max(batch).item() + 1
        #N = x.size(0)/batch_num
        #print(N)
        x1 = self.dg1(x, pos, batch)
        x2 = self.dg2(x1, pos, batch)
        #x3 = self.dg3(x2, pos, batch)
        #x3 = self.mlp(x2)
        global_fea = global_max_pool(x2, batch)
        #global_fea = global_fea.view(-1,1,256).repeat(1,N,1)
        #global_fea = global_fea.view(-1,256).contiguous()
        global_fea = torch.repeat_interleave(global_fea,1024,dim=0)
        #x = self.pool(F.relu(self.conv3(x)))
        #print(x.size())
        x4 = torch.cat([x1,x2,global_fea],dim=-1)
        fea = self.head(x4)
        score = self.activ(fea)
        return x4,score

class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=128):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_CONV(dim_feat + 128, [256, 128])
        self.mlp_3 = MLP_CONV(dim_feat + 128, [128, 128])
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat(1, 1,x1.size(2))], -2)) # (b, 256, 128)
        x3 = self.mlp_3(torch.cat([x1, feat.repeat(1, 1,x1.size(2))], -2))  # (b, 128, 256)
        #x3 = x3.permute(0, 2, 1).contiguous()
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion  # (b, 256, 3)

class PointDecoder(torch.nn.Module):
    '''
    '''
    def __init__(self,upscale):
        super(PointDecoder,self).__init__()
        self.upscale = upscale
        self.dg1 = DGCNN(MLP([3*2, 32, 64]))
        self.ps = nn.ConvTranspose1d(128, 128, upscale,upscale, bias=False)   # point-wise splitting
        self.mlp_1 = MLP_CONV(512 + 64, [256, 128])
        self.mlp_2 = MLP_CONV(128 , [64 , 3])
        
    def forward(self,pt,shapecode):
        '''
        pt: points from previous layer
        fea: global_fea
        '''
        # local_fea: batch by 512 by 1
        B,C,N = pt.size()
        # print(f"Inside PointDecoder: pt:{pt.shape}, shapecode: {shapecode.shape}")
        #print(patch_encoded[1].size())
        shapecode = shapecode.view(B,-1,1).repeat(1,1,pt.size(2))
        # print(f"Inside PointDecoder: After view : pt:{pt.shape}, shapecode: {shapecode.shape}")

        #print(pt.size())
        
        # pt: Batch by 3 by Number
        # flatten pt
        # pos: batch * number by 3
        pos = pt.permute(0,2,1).contiguous()
        # print(f"Inside PointDecoder: After permute: pos:{pos.shape}, shapecode: {shapecode.shape}")
        pos = pos.view(-1,3)
        # print(f"Inside PointDecoder: After view: pos:{pos.shape}, shapecode: {shapecode.shape}")

        
            #build batch_index
        patch_vec = torch.arange(B,dtype=torch.int64).view(-1,1)
        # print(f"Inside PointDecoder: patch_vec.shape:{patch_vec.shape}, shapecode: {shapecode.shape}")

        patch_vec = patch_vec.repeat(1,N)
        # print(f"Inside PointDecoder: After repeat: patch_vec.shape:{patch_vec.shape}, shapecode: {shapecode.shape}")

        batch = patch_vec.view(-1).to('cuda')
        # print(f"Inside PointDecoder: After view: batch.shape:{batch.shape}, shapecode: {shapecode.shape}")

        # dgfea = Batch * Number by 64
        dgfea = self.dg1(pos, pos, batch)
        # print(f"Inside PointDecoder: dg1: dgfea.shape:{dgfea.shape}, shapecode: {shapecode.shape}")

        # rel_fea: Batch by 64 by Number
        rel_fea = dgfea.view(B,-1,64).permute(0,2,1)
        # print(f"Inside PointDecoder: After view: rel_fea.shape:{rel_fea.shape}, shapecode: {shapecode.shape}")

        point_fea = torch.cat((shapecode, rel_fea), -2)
        # print(f"Inside PointDecoder: After Cat: point_fea.shape:{point_fea.shape}, shapecode: {shapecode.shape}")

        x1 = self.mlp_1(point_fea) # B by 128 by N
        # print(f"Inside PointDecoder: mlp1: x1.shape:{x1.shape}, shapecode: {shapecode.shape}")

        x_expand = self.ps(x1) # B by 128 by N*upscale 
        # print(f"Inside PointDecoder: ConvTranspose: x_expand.shape:{x_expand.shape}, shapecode: {shapecode.shape}")
        out = self.mlp_2(x_expand)  # B by 3 by N*upscale
        # print(f"Inside PointDecoder: mlp2: out.shape:{out.shape}, shapecode: {shapecode.shape}")

        out = out+torch.repeat_interleave(pt, self.upscale, dim=2)
        # rescale xyz for each patch
        #torch.sigmoid(out)
        # print(f"Inside PointDecoder: After Interleave: out.shape:{out.shape}, shapecode: {shapecode.shape}")

        return out   
class PointDecoderDoppler(torch.nn.Module):
    '''
    '''
    def __init__(self,upscale):
        super(PointDecoderDoppler,self).__init__()
        self.upscale = upscale
        self.dg1 = DGCNN(MLP([1*2, 32, 64]))
        self.ps = nn.ConvTranspose1d(128, 128, upscale,upscale, bias=False)   # point-wise splitting
        self.mlp_1 = MLP_CONV(512 + 64, [256, 128])
        self.mlp_2 = MLP_CONV(128 , [64 , 1])
        
    def forward(self,pt,shapecode):
        '''
        pt: points from previous layer
        fea: global_fea
        '''
        # local_fea: batch by 512 by 1
        B,C,N = pt.size()
        # print(f"Inside PointDecoderDoppler: pt:{pt.shape}, shapecode: {shapecode.shape}")
        #print(patch_encoded[1].size())
        shapecode = shapecode.view(B,-1,1).repeat(1,1,pt.size(2))
        # print(f"Inside PointDecoderDoppler: After view : pt:{pt.shape}, shapecode: {shapecode.shape}")
        #print(pt.size())
        # pt: Batch by 3 by Number
        # flatten pt
        # pos: batch * number by 3
        pos = pt.permute(0,2,1).contiguous()
        # print(f"Inside PointDecoderDoppler: After permute: pos:{pos.shape}, shapecode: {shapecode.shape}")
        pos = pos.view(-1,1)
        # print(f"Inside PointDecoderDoppler: After view: pos:{pos.shape}, shapecode: {shapecode.shape}")

        #build batch_index
        patch_vec = torch.arange(B,dtype=torch.int64).view(-1,1)
        # print(f"Inside PointDecoderDoppler: patch_vec.shape:{patch_vec.shape}, shapecode: {shapecode.shape}")
        patch_vec = patch_vec.repeat(1,N)
        # print(f"Inside PointDecoderDoppler: After repeat: patch_vec.shape:{patch_vec.shape}, shapecode: {shapecode.shape}")
        batch = patch_vec.view(-1).to('cuda')
        # print(f"Inside PointDecoderDoppler: After view: batch.shape:{batch.shape}, shapecode: {shapecode.shape}")
        # dgfea = Batch * Number by 64
        dgfea = self.dg1(pos, pos, batch)
        # print(f"Inside PointDecoderDoppler: dg1: dgfea.shape:{dgfea.shape}, shapecode: {shapecode.shape}")
        # rel_fea: Batch by 64 by Number
        rel_fea = dgfea.view(B,-1,64).permute(0,2,1)
        # print(f"Inside PointDecoderDoppler: After view: rel_fea.shape:{rel_fea.shape}, shapecode: {shapecode.shape}")
        point_fea = torch.cat((shapecode, rel_fea), -2)
        # print(f"Inside PointDecoderDoppler: After Cat: point_fea.shape:{point_fea.shape}, shapecode: {shapecode.shape}")
        x1 = self.mlp_1(point_fea) # B by 128 by N
        # print(f"Inside PointDecoderDoppler: mlp1: x1.shape:{x1.shape}, shapecode: {shapecode.shape}")
        x_expand = self.ps(x1) # B by 128 by N*upscale 
        # print(f"Inside PointDecoderDoppler: ConvTranspose: x_expand.shape:{x_expand.shape}, shapecode: {shapecode.shape}")
        out = self.mlp_2(x_expand)  # B by 3 by N*upscale
        # print(f"Inside PointDecoderDoppler: mlp2: out.shape:{out.shape}, shapecode: {shapecode.shape}")
        out = out+torch.repeat_interleave(pt, self.upscale, dim=2)
        # rescale xyz for each patch
        #torch.sigmoid(out)
        # print(f"Inside PointDecoderDoppler: After Interleave: out.shape:{out.shape}, shapecode: {shapecode.shape}")
        return out   



class DownsamplePoints(nn.Module):
    def __init__(self, num_input_points=16384, num_output_points=1024):
        super(DownsamplePoints, self).__init__()
        self.num_input_points = num_input_points
        self.num_output_points = num_output_points
        # Downsampling layer for pd_points
        self.pd_points_downsample = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=512, stride=2)

        self.fc = nn.Linear(1601 * 3, 1024 * 3)

    def forward(self, score, pd_points, ini_points):
        """
        score: [16384, 1]
        pd_points: [16, 16384, 3]
        ini_points: [16, 64, 3] (unchanged)
        """
        batch_size = pd_points.shape[0]
        # print("Batch Size: ", batch_size)
        # print("Inside Downsample: ", "Before pd_points_downsample pd_points.shape: ", pd_points.shape)
        pd_points = pd_points.permute(0, 2, 1)
        # print("Inside Downsample: ", "After permute_1 pd_points.shape: ", pd_points.shape)
        pd_points = self.pd_points_downsample(pd_points)
        # print("Inside Downsample: ", "After pd_points_downsample_1 pd_points.shape: ", pd_points.shape)
        pd_points = self.pd_points_downsample(pd_points)
        # print("Inside Downsample: ", "After pd_points_downsample_2 pd_points.shape: ", pd_points.shape)
        pd_points = self.pd_points_downsample(pd_points)
        # print("Inside Downsample: ", "After pd_points_downsample_3 pd_points.shape: ", pd_points.shape)
        pd_points = pd_points.view(batch_size, -1)
        # print("Inside Downsample: ", "After multiplication axis 3 pd_points.shape: ", pd_points.shape)
        pd_points = self.fc(pd_points)  # [16, 1024 * 3]
        # print("Inside Downsample: ", "After dense layeer pd_points.shape: ", pd_points.shape)
        pd_points = pd_points.view(batch_size, 3, 1024)
        # print("Inside Downsample: ", "After final reshape pd_points.shape: ", pd_points.shape)
        pd_points = pd_points.permute(0, 2, 1)
        # print("Inside Downsample: ", "After permute_2 pd_points.shape: ", pd_points.shape)


        return score, pd_points, ini_points  
    
    
class DownsampleDoppler(nn.Module):
    def __init__(self, num_input_points=16384, num_output_points=1024):
        super(DownsampleDoppler, self).__init__()
        self.num_input_points = num_input_points
        self.num_output_points = num_output_points
        # Downsampling layer for pd_points
        self.pd_Doppler_downsample = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=512, stride=2)

        self.fc = nn.Linear(1601 * 1, 1024 * 1)

    def forward(self, score, pd_Doppler, ini_points):
        """
        score: [16384, 1]
        pd_Doppler: [16, 16384, 1]
        ini_points: [16, 64, 3] (unchanged)
        """
        batch_size = pd_Doppler.shape[0]
        print("Batch Size: ", batch_size)
        print("Inside Downsample: ", "Before pd_Doppler_downsample pd_Doppler.shape: ", pd_Doppler.shape)
        pd_Doppler = pd_Doppler.permute(0, 2, 1)
        print("Inside Downsample: ", "After permute_1 pd_Doppler.shape: ", pd_Doppler.shape)
        pd_Doppler = self.pd_Doppler_downsample(pd_Doppler)
        # print("Inside Downsample: ", "After pd_Doppler_downsample_1 pd_Doppler.shape: ", pd_Doppler.shape)
        pd_Doppler = self.pd_Doppler_downsample(pd_Doppler)
        # print("Inside Downsample: ", "After pd_Doppler_downsample_2 pd_Doppler.shape: ", pd_Doppler.shape)
        pd_Doppler = self.pd_Doppler_downsample(pd_Doppler)
        # print("Inside Downsample: ", "After pd_Doppler_downsample_3 pd_Doppler.shape: ", pd_Doppler.shape)
        pd_Doppler = pd_Doppler.view(batch_size, -1)
        # print("Inside Downsample: ", "After multiplication axis 3 pd_Doppler.shape: ", pd_Doppler.shape)
        pd_Doppler = self.fc(pd_Doppler)  # [16, 1024 * 3]
        # print("Inside Downsample: ", "After dense layeer pd_Doppler.shape: ", pd_Doppler.shape)
        pd_Doppler = pd_Doppler.view(batch_size, 1, 1024)
        # print("Inside Downsample: ", "After final reshape pd_Doppler.shape: ", pd_Doppler.shape)
        pd_Doppler = pd_Doppler.permute(0, 2, 1)
        # print("Inside Downsample: ", "After permute_2 pd_Doppler.shape: ", pd_Doppler.shape)


        return score, pd_Doppler, ini_points  
    

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.score_estimator = NoiseScoreNet()
        self.shape_extractor = MMGraphExtractor()
        self.seed_gen = SeedGenerator(dim_feat=512, num_pc=64)
        self.decoder2 = PointDecoder(upscale=8)
        self.decoder = PointDecoder(upscale=4)
        self.decoder4 = PointDecoderDoppler(upscale=8)
        self.decoder3 = PointDecoderDoppler(upscale=2)

        # self.encoderscore_estimator = NoiseScoreNetEncoder()
        # self.encodershape_extractor = MMGraphGenerator()
        # self.encoderseed_gen = SeedCompressor(dim_feat=512, num_pc=64)
        # self.encoder2 = PointEncoder(downscale=4)
        # self.encoder = PointEncoder(downscale=8)

        self.pdPointsDownsample = DownsamplePoints()
        self.pdDopplerDownsample = DownsampleDoppler()

    def forward(self,x_ini, x_pos, x_pi):

        #####Decoder###########
        '''
        x_ini.shape:  torch.Size([batchSize, 3]) x_pos.shape:  torch.Size([16384, 3]) x_pi.shape:  torch.Size([16384])
        x: the feature of original incomplete color pcd
            : conditional feature
        y: complete pcd without color
            : pos
        '''
        print("Input: ", "x_ini.shape: ", x_ini.shape, "x_pos.shape: ",x_pos.shape, "x_pi.shape: ", x_pi.shape)
        ## feature extractor
        batch_num = torch.max(x_pi) + 1
        print("batch_num.shape: ", batch_num)
        #x_fea = x_fea.T.contiguous() 
        x_fea, score = self.score_estimator(x_pos.squeeze(),x_pos.squeeze(), x_pi)
        print("score_estimator: ", "x_fea.shape: ",x_fea.shape,"score.shape: ",score.shape)
        shape_fea = self.shape_extractor(x_fea, x_pos, score.squeeze(), x_pi)
        print("After  shape_extractor : shape_fea.shape: ",shape_fea.shape)
        ## point cloud reconstruction
        shape_fea = shape_fea.view(batch_num,-1,1)
        print("Afer view shape_fea.shape: ",shape_fea.shape)
        init_point = self.seed_gen(shape_fea)
        print("SeedGen: ","ini_points: ", init_point.shape)
        print("===============================================================")
        pd_point = self.decoder(init_point,shape_fea)
        print("After Decoder: ", "pd_points: ", pd_point.shape, "ini_points: ", init_point.shape)
        print("===============================================================")
        pd_point = self.decoder2(pd_point,shape_fea)
        print("After Decoder2_1: ", "pd_points: ", pd_point.shape, "ini_points: ", init_point.shape)
        print("===============================================================")
        pd_point = self.decoder2(pd_point,shape_fea)
        print("After Decoder2_2: ", "pd_points: ", pd_point.shape, "ini_points: ", init_point.shape)
        print("===============================================================")
        print(f"Doppler: After view ini_doppler.shape: {x_ini.shape}")
        ini_doppler = x_ini.view(batch_num,-1,1)
        print(f"Doppler: After view ini_doppler.shape: {ini_doppler.shape}")
        ini_doppler = ini_doppler.permute(0,2,1)
        print(f"Doppler: After permute ini_doppler.shape: {ini_doppler.shape}")
        print("===============================================================")
        pd_doppler = self.decoder3(ini_doppler,shape_fea)
        print(f"After Doppler Decoder3_1: pd_doppler.shape: {pd_doppler.shape}, ini_doppler: {ini_doppler.shape})")        
        print("===============================================================")
        pd_doppler = self.decoder4(pd_doppler,shape_fea)
        print(f"After Doppler Decoder3_2: pd_doppler.shape: {pd_doppler.shape}, ini_doppler: {ini_doppler.shape})")
        print("===============================================================")
        #x_ini_coarse = x_ini.view(-1,1,3).repeat(1,init_point.size(2),1)
        #ini_points = init_point.view(batch_num,-1,3)
        ini_points = init_point.permute(0, 2, 1).contiguous()
        pd_points = pd_point.permute(0, 2, 1).contiguous()
        pd_doppler = pd_doppler.permute(0, 2, 1).contiguous()
        print("After Permute: ", "score.shape: ", score.shape, "pd_points: ", pd_points.shape, "ini_points: ", ini_points.shape)
        scoreEncoded,pd_pointsEncoded,ini_pointsEncoded = self.pdPointsDownsample(score,pd_points,ini_points)
        print("After pdPointsDownsample: ", "scoreEncoded.shape: ",scoreEncoded.shape, "pd_pointsEncoded: ", pd_pointsEncoded.shape, "ini_pointsEncoded: ", ini_pointsEncoded.shape)
        print("===============================================================")
        scoreEncoded,pd_DopplerEncoded,ini_pointsEncoded = self.pdDopplerDownsample(score,pd_doppler,ini_points)
        print("After pdPointsDownsample: ", "scoreEncoded.shape: ",scoreEncoded.shape, "pd_DopplerEncoded: ", pd_DopplerEncoded.shape, "ini_pointsEncoded: ", ini_pointsEncoded.shape)
        return score,ini_points,pd_points,pd_doppler,scoreEncoded,pd_pointsEncoded,pd_DopplerEncoded,ini_pointsEncoded

        
if __name__ == '__main__':
    print("Hi")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Autoencoder().to(device)
    G.zero_grad()
    # print(summary(G, input_data=input_shapes, col_names=["input_size", "output_size", "num_params", "trainable"], depth=3))
    x_ini = torch.rand(64, 1024).to(device)
    x_pos = torch.rand(65536, 3).to(device)
    x_batch = torch.randint(0, 64, (65536,)).to(device)
    input_shapes = (x_ini, x_pos, x_batch)

    summary_file_path = "model_summaryAutoencoder.txt"

    with open(summary_file_path, "w") as f:
        with redirect_stdout(f):
            print(summary(
                G, 
                input_data=input_shapes, 
                col_names=["input_size", "output_size", "num_params", "trainable"], 
                depth=3
            ))

    print(f"Model summary saved to {summary_file_path}")
    print("Parameter List:")
    for name, param in G.named_parameters():
        print(f"{name}: {'Trainable' if param.requires_grad else 'Non-trainable'} | Shape: {param.shape} | Numel: {param.numel()}")
    print("Succesfully run the model file")
import torch
import torch.nn as nn

from Emd.emd_module import emdFunction
# from chamferdist import ChamferDistance#installed as packages
from chamfer_distance import ChamferDistance#local file
def compute_confidence_score(input_pcd, gt_pcd):
    batch_size, num_points, _ = input_pcd.shape
    _, gt_num_points, _ = gt_pcd.shape
    input_pcd_exp = input_pcd.unsqueeze(2).repeat(1, 1, gt_num_points, 1)  #[32, 1000, N, 3]
    gt_pcd_exp = gt_pcd.unsqueeze(1).repeat(1, num_points, 1, 1)  #[32, 1000, N, 3]
    distances = torch.norm(input_pcd_exp - gt_pcd_exp, dim=-1)  #[32, 1000, N]
    min_distances, _ = torch.min(distances, dim=-1)  #[32, 1000]
    ground_truth_scores = torch.exp(-min_distances)  #[32, 1000]
    return ground_truth_scores.unsqueeze(-1)  #[32, 1000, 1]

def emd(p1,p2):
    emdist, _ = emdFunction.apply(p1, p2, 0.01, 500)
    return torch.sqrt(emdist).mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5,beta=0.5):
        super(CombinedLoss, self).__init__()
        # self.chamfer = ChamferDistance()
        self.ChD = ChamferDistance()
        # self.emd = emd()

        self.mse = nn.MSELoss()
        self.mse2=nn.MSELoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pc1, pc2, pc3):
        '''
            pc1-> referecne to model weights
            pc2-> ground truth pcd
            pc3->radarpcd
        '''     
        # print(pc1[3].shape,pc2.shape)
        # emd_dist = self.emd(pc1[0], pc2)
        # print(emd_dist)
        # print("emd_dist.shape: ",emd_dist.shape)
        # # emd_dist, _ = emd_loss(pc1[0], pc2, eps=0.005, max_iters=1000)
        # emd_dist = pairwise_distances(pc1[0], pc2)

        ######################################################
        ground_truth_scores = compute_confidence_score(pc3, pc2)  # Output shape: [32, 1000, 1]
        confidenseScoreLoss = self.mse(pc1[3],ground_truth_scores)
        print(pc1[1].shape,pc2.shape)
        dist3, dist4, idx3, idx4 = self.ChD(pc1[1],pc2)
        dist1, dist2, idx1, idx2 = self.ChD(pc1[0], pc2)
        chamferLossUp  = 0.5*(torch.mean(dist1)) + 2*(torch.mean(dist2))#torch.mean(dist1)
        chamferLossSeed  = torch.mean(dist3)
        seedGeneratorLossCHD = 1*chamferLossSeed
        upSamplingBlockLossCHD = self.alpha*chamferLossUp
        upSamplingBlockLossEMD = emd(pc1[0], pc2)
        # return seedGeneratorLossCHD
        # finalLoss = upSamplingBlockLossCHD + self.beta*seedGeneratorLoss + self.alpha*confidenseScoreLoss +  upSamplingBlockLossEMD
        finalLoss = upSamplingBlockLossCHD + self.alpha*confidenseScoreLoss +  upSamplingBlockLossEMD
        ########################################################
        # print(pc1[0].shape, pc2.shape)
        # finalLoss = emd(pc1[0], pc2)#self.ChD(pc1[0], pc2)#using custom class
        # finalLoss = compute_emd_loss(pc1[0], pc2)#usign custom function
        return finalLoss

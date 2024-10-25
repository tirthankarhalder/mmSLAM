import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree
# from pyemd import emd
# from pytorch3d.loss import emd_loss
from scipy.spatial import cKDTree
from pyemd import emd
# from concurrent.futures import ThreadPoolExecutor, as_completed


def compute_confidence_score(input_pcd, gt_pcd):
    batch_size, num_points, _ = input_pcd.shape
    _, gt_num_points, _ = gt_pcd.shape
    input_pcd_exp = input_pcd.unsqueeze(2).repeat(1, 1, gt_num_points, 1)  #[32, 1000, N, 3]
    gt_pcd_exp = gt_pcd.unsqueeze(1).repeat(1, num_points, 1, 1)  #[32, 1000, N, 3]
    distances = torch.norm(input_pcd_exp - gt_pcd_exp, dim=-1)  #[32, 1000, N]
    min_distances, _ = torch.min(distances, dim=-1)  #[32, 1000]
    ground_truth_scores = torch.exp(-min_distances)  #[32, 1000]
    return ground_truth_scores.unsqueeze(-1)  #[32, 1000, 1]

def pairwise_distances(x, y):
    x_square = torch.sum(x ** 2, dim=-1, keepdim=True)  #[batch_size, num_points, 1]
    y_square = torch.sum(y ** 2, dim=-1, keepdim=True).transpose(2, 1)  #[batch_size, 1, num_points]
    distances = x_square + y_square - 2 * torch.bmm(x, y.transpose(2, 1))  #[batch_size, num_points, num_points]
    distances = torch.clamp(distances, min=0.0)
    return torch.sqrt(distances)

def earth_mover_distance(S1, S2):
    pairwise_dist = pairwise_distances(S1, S2)  # Shape: [batch_size, num_points, num_points]
    min_distances, _ = torch.min(pairwise_dist, dim=-1)  # Shape: [batch_size, num_points]
    emd = torch.mean(min_distances, dim=-1)  # Shape: [batch_size]
    return emd.mean()
def compute_pairwise_distances(x, y):
    # x: [n, d]  y: [m, d]
    diff = np.expand_dims(x, axis=1) - np.expand_dims(y, axis=0)
    dist = np.linalg.norm(diff, axis=-1)
    return dist
# Earth Mover's Distance (EMD) for one batch (source_points[0] and target_points[0])
def earth_movers_distance(source, target):
    # Compute the cost matrix (pairwise distances between points)
    dist_matrix = compute_pairwise_distances(source, target)

    # Uniform weights for source and target points
    source_weights = np.ones(source.shape[0]) / source.shape[0]
    target_weights = np.ones(target.shape[0]) / target.shape[0]

    # Compute EMD
    emd_distance = emd(source_weights, target_weights, dist_matrix)

    return emd_distance

# class ChamferDistance(torch.nn.Module):
#     def __init__(self):
#         super(ChamferDistance, self).__init__()
#         self.batch_size = 32
#     def process_pc2(self, tree1, pc2, dist1_list):
#         for i in range(0, len(pc2), self.batch_size):
#             # print(f"pc2 {i}")
#             pc2_batch = pc2[i:i + self.batch_size]
#             pc2_batch_cpu = pc2_batch.cpu().detach().numpy()
#             dist1, _ = tree1.query(pc2_batch_cpu, k=1)
#             dist1_list.append(np.mean(dist1))
#         return dist1_list

#     def process_pc1(self, tree2, pc1, dist2_list):
#         for i in range(0, len(pc1), self.batch_size):
#             # print(f"pc1 {i}")
#             pc1_batch = pc1[i:i + self.batch_size]
#             pc1_batch_cpu = pc1_batch.cpu().detach().numpy()
#             dist2, _ = tree2.query(pc1_batch_cpu, k=1)
#             dist2_list.append(np.mean(dist2))
#         return dist2_list
#     def forward(self, pc1, pc2):
#         reshapedpc1 = pc1.reshape(-1,3)
#         reshapedpc2 = pc2.reshape(-1,3)
#         pc1 = reshapedpc1.cpu().detach().numpy()
#         pc2 = reshapedpc2.cpu().detach().numpy()
#         tree1 = cKDTree(pc1)
#         tree2 = cKDTree(pc2)
#         dist1_list = []
#         dist2_list = []

#         # print(len(pc1))
#         # print(len(pc2))
#         # # Compute chamfer distance
#         # with ThreadPoolExecutor(max_workers=20) as executor:
#         #     future_pc2 = executor.submit(self.process_pc2, tree1, reshapedpc2, dist1_list)
#         #     future_pc1 = executor.submit(self.process_pc1, tree2, reshapedpc1, dist2_list)
#         #     # Wait for both futures to complete
#         #     dist1_list = future_pc2.result()
#         #     dist2_list = future_pc1.result()
#         # chamfer_dist = np.mean(dist1_list) + np.mean(dist2_list)

#         # return torch.tensor(chamfer_dist, dtype=torch.float32).to(pc1.device)
#         dist1, _ = tree1.query(pc2, k=1)
#         dist2, _ = tree2.query(pc1, k=1)
#         chamfer_dist = np.mean(dist1) + np.mean(dist2)
#         return torch.tensor(chamfer_dist, dtype=torch.float32).to('cpu')


class ChamferDistance(torch.nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, pc1, pc2):
        reshapedpc1 = pc1.reshape(-1,3)
        reshapedpc2 = pc2.reshape(-1,3)
        pc1 = reshapedpc1.cpu().detach().numpy()
        pc2 = reshapedpc2.cpu().detach().numpy()
        tree1 = cKDTree(pc1)
        tree2 = cKDTree(pc2)
        dist1, _ = tree1.query(pc2, k=1)
        dist2, _ = tree2.query(pc1, k=1)
        chamfer_dist = np.mean(dist1) + np.mean(dist2)
        return torch.tensor(chamfer_dist, dtype=torch.float32).to('cpu')


class EarthMoversDistanceOpend3d(torch.nn.Module):
    def __init__(self):
        super(EarthMoversDistanceOpend3d,self).__init__()

    def forward(self,pc1,pc2):
        pc1 = pc1[0].cpu().detach().numpy()
        pc2 = pc2[0].cpu().detach().numpy()
        diff = np.expand_dims(pc1, axis=1) - np.expand_dims(pc2, axis=0)
        dist_matrix = np.linalg.norm(diff, axis=-1)
        source_weights = np.ones(pc1.shape[0]) / pc1.shape[0]
        target_weights = np.ones(pc2.shape[0]) / pc2.shape[0]
        emd_distance = emd(source_weights.astype('float64'), target_weights.astype('float64'), dist_matrix.astype('float64'))
        return emd_distance

class EarthMoversDistance(torch.nn.Module):##this has some memory problem
    def __init__(self):
        super(EarthMoversDistance, self).__init__()

    def forward(self, pc1, pc2):
        print("Inisde EMD",pc1.shape,pc2.shape)
        reshapedpc1 = pc1.reshape(-1,3)
        reshapedpc2 = pc2.reshape(-1,3)
        pc1 = reshapedpc1.cpu().detach().numpy()
        pc2 = reshapedpc2.cpu().detach().numpy()

        # pc1 = pc1.cpu().detach().numpy()
        # pc2 = pc2.cpu().detach().numpy()

        print(pc1.shape,pc2.shape)

        if len(pc1) != len(pc2):
            raise ValueError("Point clouds must have the same number of points for EMD computation.")
        dists = np.linalg.norm(pc1[:, np.newaxis] - pc2[np.newaxis, :], axis=2)
        weights = np.ones(len(pc1)) / len(pc1)
        emd_distance = emd(weights, weights, dists)
        return torch.tensor(emd_distance, dtype=torch.float32).to(pc1.device)

class EarthMoversDistanceBatchWise(torch.nn.Module):#emd with batch wise operation
    def __init__(self):
        super(EarthMoversDistance, self).__init__()

    def forward(self, pc1, pc2):
        batch_size = pc1.shape[0]
        emd_distances = []
        for i in range(batch_size):
            reshapedpc1 = pc1[i].reshape(-1, 3).double().cpu().detach().numpy()
            reshapedpc2 = pc2[i].reshape(-1, 3).double().cpu().detach().numpy()
            if len(reshapedpc1) != len(reshapedpc2):
                raise ValueError("Point clouds must have the same number of points for EMD computation.")
            dists = np.linalg.norm(reshapedpc1[:, np.newaxis] - reshapedpc2[np.newaxis, :], axis=2)
            weights = np.ones(len(reshapedpc1)) / len(reshapedpc1)
            emd_distance = emd(weights, weights, dists)
            emd_distances.append(torch.tensor(emd_distance, dtype=torch.float32))
        emd_distances =  torch.stack(emd_distances).float().to(pc1.device)
        return emd_distances.sum()


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.chamfer = ChamferDistance()
        self.emd = EarthMoversDistance()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pc1, pc2, pc3):
        '''
            pc1-> referecne to model weights
            pc2-> ground truth pcd
            pc3->radarpcd
        '''

        cd = self.chamfer(pc1[0], pc2)
        # print(pc1[3].shape,pc2.shape)

        emd_dist = self.emd(pc1[0], pc2)
        print("emd_dist.shape: ",emd_dist.shape)
        # emd_dist, _ = emd_loss(pc1[0], pc2, eps=0.005, max_iters=1000)
        emd_dist = pairwise_distances(pc1[0], pc2)

        ground_truth_scores = compute_confidence_score(pc3, pc2)  # Output shape: [32, 1000, 1]

        confidenseScoreLoss = self.mse(pc1[3],ground_truth_scores)

        seedGeneratorLoss = self.chamfer(pc1[1],pc2)

        upSamplingBlockLoss = self.alpha * cd + (1 - self.alpha) * emd_dist

        finalLoss = confidenseScoreLoss + upSamplingBlockLoss + seedGeneratorLoss
        return finalLoss






# if __name__=="__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = YourModel().to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     criterion = CombinedLoss(alpha=0.5).to(device)

#     for epoch in range(num_epochs):
#         model.train()
#         for data in dataloader:
#             inputs, targets = data
#             inputs, targets = inputs.to(device), targets.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)

#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()

#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')














# import torch
# import torch.nn as nn
# import numpy as np
# from scipy.spatial import cKDTree
# # from pyemd import emd
# # from pytorch3d.loss import emd_loss
# from scipy.spatial import cKDTree
# from pyemd import emd
# from chamfer_distance import ChamferDistance as chamfer_dist
# from multiprocessing import Pool



# def compute_confidence_score(input_pcd, gt_pcd):
#     input_pcd = input_pcd.cpu()
#     gt_pcd = gt_pcd.cpu()
#     batch_size, num_points, _ = input_pcd.shape
#     _, gt_num_points, _ = gt_pcd.shape
#     input_pcd_exp = input_pcd.unsqueeze(2).repeat(1, 1, gt_num_points, 1)  #[32, 1000, N, 3]
#     gt_pcd_exp = gt_pcd.unsqueeze(1).repeat(1, num_points, 1, 1)  #[32, 1000, N, 3]
#     distances = torch.norm(input_pcd_exp - gt_pcd_exp, dim=-1)  #[32, 1000, N]
#     min_distances, _ = torch.min(distances, dim=-1)  #[32, 1000]
#     ground_truth_scores = torch.exp(-min_distances)  #[32, 1000]
#     return ground_truth_scores.unsqueeze(-1)  #[32, 1000, 1]

# def pairwise_distances(x, y):
#     x_square = torch.sum(x ** 2, dim=-1, keepdim=True)  #[batch_size, num_points, 1]
#     y_square = torch.sum(y ** 2, dim=-1, keepdim=True).transpose(2, 1)  #[batch_size, 1, num_points]
#     distances = x_square + y_square - 2 * torch.bmm(x, y.transpose(2, 1))  #[batch_size, num_points, num_points]
#     distances = torch.clamp(distances, min=0.0)
#     return torch.sqrt(distances)

# def earth_mover_distance(S1, S2):
#     pairwise_dist = pairwise_distances(S1, S2)  # Shape: [batch_size, num_points, num_points]
#     min_distances, _ = torch.min(pairwise_dist, dim=-1)  # Shape: [batch_size, num_points]
#     emd = torch.mean(min_distances, dim=-1)  # Shape: [batch_size]
#     return emd.mean()

# def compute_chamfer_distance(point_cloud1, point_cloud2):
#     print('Inside compute chamfer')
#     # point_cloud1, point_cloud2 = pair
#     point_cloud1 = point_cloud1.detach().cpu()
#     point_cloud2 = point_cloud2.detach().cpu()
#     print('starting chamfer compute')
#     # Replace with your Chamfer distance calculation
#     chd = chamfer_dist()
#     dist1, dist2, _, _ = chd(point_cloud1.unsqueeze(0), point_cloud2.unsqueeze(0))
#     print('done compute')
#     loss = (torch.mean(dist1)) + (torch.mean(dist2))
#     print('loss returned')
#     return loss

# class ChamferDistance(torch.nn.Module):
#     def __init__(self, point_chunk_size=64):
#         super(ChamferDistance, self).__init__()
#         self.point_chunk_size = point_chunk_size  # Control the chunk size to manage memory

#     def forward(self, pred_points, gt_points):
#         return compute_chamfer_distance(pred_points,gt_points)
#         # pred_points: [batch_size, N1, 3]
#         # gt_points: [batch_size, N2, 3]
#         # pred_points = pred_points.cpu() 
#         # gt_points = gt_points.cpu()
#         # batch_size, N1, _ = pred_points.size()
#         # _, N2, _ = gt_points.size()
        
#         # chamfer_loss = 0.0

#         # Iterate over chunks of pred_points
#         # for i in range(0, N1, self.chunk_size):
#         #     print(f"Inside Pred {i}")
#         #     pred_chunk = pred_points[:, i:i+self.chunk_size, :]  # Take chunks of pred_points

#         #     # Expand dims for broadcasting
#         #     pred_chunk_exp = pred_chunk.unsqueeze(2)  # [batch_size, chunk_size, 1, 3]
#         #     gt_points_exp = gt_points.unsqueeze(1)    # [batch_size, 1, N2, 3]
            
#         #     # Compute pairwise distances (squared L2-norm) for the chunk
#         #     dist_chunk = torch.sum((pred_chunk_exp - gt_points_exp) ** 2, dim=-1)  # [batch_size, chunk_size, N2]
            
#         #     # Find the minimum distance from each pred_point in chunk to any point in gt_points
#         #     min_dist_S1_to_S2, _ = torch.min(dist_chunk, dim=2)  # [batch_size, chunk_size]
#         #     chamfer_loss += torch.sum(min_dist_S1_to_S2)  # Accumulate the loss

#         # # Iterate over chunks of gt_points to compute the reverse direction (S2 to S1)
#         # for j in range(0, N2, self.chunk_size):
#         #     print(f"Inside Pred {j}")
#         #     gt_chunk = gt_points[:, j:j+self.chunk_size, :]  # Take chunks of gt_points

#         #     # Expand dims for broadcasting
#         #     pred_points_exp = pred_points.unsqueeze(2)   # [batch_size, N1, 1, 3]
#         #     gt_chunk_exp = gt_chunk.unsqueeze(1)         # [batch_size, 1, chunk_size, 3]
            
#         #     # Compute pairwise distances (squared L2-norm) for the chunk
#         #     dist_chunk = torch.sum((pred_points_exp - gt_chunk_exp) ** 2, dim=-1)  # [batch_size, N1, chunk_size]
            
#         #     # Find the minimum distance from each gt_point in chunk to any point in pred_points
#         #     min_dist_S2_to_S1, _ = torch.min(dist_chunk, dim=1)  # [batch_size, chunk_size]
#         #     chamfer_loss += torch.sum(min_dist_S2_to_S1)  # Accumulate the loss
#         # for i in range(0, N1, self.point_chunk_size):
#         #     pred_chunk = pred_points[:, i:i+self.point_chunk_size, :]  # Take smaller point batches from pred_points
#         #     print(f"Inside i {i}")
#         #     # Compute the minimum distance for this chunk
#         #     for j in range(0, N2, self.point_chunk_size):
#         #         print(f"Inside i {i}")
#         #         gt_chunk = gt_points[:, j:j+self.point_chunk_size, :]  # Take smaller point batches from gt_points

#         #         pred_chunk_exp = pred_chunk.unsqueeze(2)  # [batch_size, chunk_size, 1, 3]
#         #         gt_chunk_exp = gt_chunk.unsqueeze(1)      # [batch_size, 1, chunk_size, 3]
                
#         #         # Compute pairwise distances (L2 norm) between these small chunks
#         #         dist_chunk = torch.sum((pred_chunk_exp - gt_chunk_exp) ** 2, dim=-1)  # [batch_size, chunk_size, chunk_size]
                
#         #         # For each point in S1, find the minimum distance to a point in S2
#         #         min_dist_S1_to_S2, _ = torch.min(dist_chunk, dim=2)  # [batch_size, chunk_size]
#         #         chamfer_loss += torch.sum(min_dist_S1_to_S2)
                
#         #         # For each point in S2, find the minimum distance to a point in S1
#         #         min_dist_S2_to_S1, _ = torch.min(dist_chunk, dim=1)  # [batch_size, chunk_size]
#         #         chamfer_loss += torch.sum(min_dist_S2_to_S1)
#         #         del min_dist_S1_to_S2
#         #         del min_dist_S2_to_S1
#         #         del dist_chunk
#         # # Normalize by the number of points
#         # chamfer_loss = chamfer_loss / (N1 + N2)

#         # return chamfer_loss / batch_size
# # def compute_pairwise_distances(x, y):
# #     # x: [n, d]  y: [m, d]
# #     diff = np.expand_dims(x, axis=1) - np.expand_dims(y, axis=0)
# #     dist = np.linalg.norm(diff, axis=-1)
# #     return dist

# # # Earth Mover's Distance (EMD) for one batch (source_points[0] and target_points[0])
# # def earth_movers_distance(source, target):
# #     # Compute the cost matrix (pairwise distances between points)
# #     dist_matrix = compute_pairwise_distances(source, target)
    
# #     # Uniform weights for source and target points
# #     source_weights = np.ones(source.shape[0]) / source.shape[0]
# #     target_weights = np.ones(target.shape[0]) / target.shape[0]
    
# #     # Compute EMD
# #     emd_distance = emd(source_weights, target_weights, dist_matrix)
    
# #     return emd_distance

# class EarthMoversDistanceOpend3d(torch.nn.Module):
#     def __init__(self):
#         super(EarthMoversDistanceOpend3d,self).__init__()

#     def forward(self,pc1,pc2):
#         pc1 = pc1[0].cpu().detach().numpy()
#         pc2 = pc2[0].cpu().detach().numpy()
#         diff = np.expand_dims(pc1, axis=1) - np.expand_dims(pc2, axis=0)
#         dist_matrix = np.linalg.norm(diff, axis=-1)
#         source_weights = np.ones(pc1.shape[0]) / pc1.shape[0]
#         target_weights = np.ones(pc2.shape[0]) / pc2.shape[0]
#         emd_distance = emd(source_weights.astype('float64'), target_weights.astype('float64'), dist_matrix.astype('float64'))
#         return emd_distance

# class EarthMoversDistance(torch.nn.Module):##this has some memory problem
#     def __init__(self):
#         super(EarthMoversDistance, self).__init__()

#     def forward(self, pc1, pc2):

#         reshapedpc1 = pc1.reshape(-1,3)
#         reshapedpc2 = pc2.reshape(-1,3)
#         pc1 = reshapedpc1.cpu().detach().numpy()
#         pc2 = reshapedpc2.cpu().detach().numpy()

#         # pc1 = pc1.cpu().detach().numpy()
#         # pc2 = pc2.cpu().detach().numpy()

#         # print(pc1.shape,pc2.shape)
#         if len(pc1) != len(pc2):
#             raise ValueError("Point clouds must have the same number of points for EMD computation.")
#         dists = np.linalg.norm(pc1[:, np.newaxis] - pc2[np.newaxis, :], axis=2)
#         weights = np.ones(len(pc1)) / len(pc1)
#         emd_distance = emd(weights, weights, dists)
#         return torch.tensor(emd_distance, dtype=torch.float32).to(pc1.device)

# # class EarthMoversDistance(torch.nn.Module):#emd with batch wise operation
# #     def __init__(self):
# #         super(EarthMoversDistance, self).__init__()

# #     def forward(self, pc1, pc2):
# #         batch_size = pc1.shape[0]
# #         emd_distances = []
# #         for i in range(batch_size):
# #             reshapedpc1 = pc1[i].reshape(-1, 3).double().cpu().detach().numpy()
# #             reshapedpc2 = pc2[i].reshape(-1, 3).double().cpu().detach().numpy()
# #             if len(reshapedpc1) != len(reshapedpc2):
# #                 raise ValueError("Point clouds must have the same number of points for EMD computation.")
# #             dists = np.linalg.norm(reshapedpc1[:, np.newaxis] - reshapedpc2[np.newaxis, :], axis=2)
# #             weights = np.ones(len(reshapedpc1)) / len(reshapedpc1)
# #             emd_distance = emd(weights, weights, dists)
# #             emd_distances.append(torch.tensor(emd_distance, dtype=torch.float32))
# #         emd_distances =  torch.stack(emd_distances).float().to(pc1.device)
# #         return emd_distances.sum()


# class CombinedLoss(nn.Module):
#     def __init__(self, alpha=0.5):
#         super(CombinedLoss, self).__init__()
#         self.chamfer = ChamferDistance()
#         self.emd = EarthMoversDistance()
#         self.mse = nn.MSELoss()
#         self.alpha = alpha

#     def forward(self, pc1, pc2, pc3):
#         '''
#             pc1-> referecne to model weights
#             pc2-> ground truth pcd
#             pc3->radarpcd
#         '''
#         print("Inside loss")
#         print(f"pc1[0]: {pc1[0].shape}, pc2.shape: {pc2.shape}")
#         cd = self.chamfer(pc1[0], pc2)
#         print("chamfer done")
#         # print(pc1[3].shape,pc2.shape)

#         # emd_dist = self.emd(pc1[0], pc2)
#         # print("emd_dist.shape: ",emd_dist.shape)
#         # emd_dist, _ = emd_loss(pc1[0], pc2, eps=0.005, max_iters=1000)
#         # emd_dist = pairwise_distances(pc1[0], pc2)

#         ground_truth_scores = compute_confidence_score(pc3, pc2)  # Output shape: [32, 1000, 1]

#         confidenseScoreLoss = self.mse(pc1[3],ground_truth_scores)

#         seedGeneratorLoss = self.chamfer(pc1[1],pc2)

#         upSamplingBlockLoss = self.alpha * cd #+ (1 - self.alpha) * emd_dist

#         finalLoss = confidenseScoreLoss + upSamplingBlockLoss + seedGeneratorLoss
#         return finalLoss
    




 
# # if __name__=="__main__":
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     model = YourModel().to(device)

# #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# #     criterion = CombinedLoss(alpha=0.5).to(device)

# #     for epoch in range(num_epochs):
# #         model.train()
# #         for data in dataloader:
# #             inputs, targets = data
# #             inputs, targets = inputs.to(device), targets.to(device)
            
# #             optimizer.zero_grad()
# #             outputs = model(inputs)
            
# #             loss = criterion(outputs, targets)
# #             loss.backward()
# #             optimizer.step()
            
# #         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

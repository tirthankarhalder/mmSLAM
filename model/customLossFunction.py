import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree
# from pytorch3d.loss import emd_loss
from scipy.spatial import cKDTree
from pyemd import emd
from scipy.optimize import linear_sum_assignment

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
            print(i)
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


class EarthMoverDistance(torch.nn.Module):
    def __init__(self):
        super(EarthMoverDistance, self).__init__()

    def forward(self, S1, S2):
        """
        Args:
            S1: Tensor of shape (batch_size, N, 3) - first point set
            S2: Tensor of shape (batch_size, N, 3) - second point set (should be same size as S1)
        
        Returns:
            emd_loss: EMD loss value
        """
        batch_size, N, _ = S1.size()
        emd_loss = 0.0

        for i in range(batch_size):
            print(i)
            # Calculate the pairwise distances between points in S1 and S2
            cost_matrix = torch.cdist(S1[i], S2[i], p=2).cpu().detach().numpy()
            
            # Solve for the minimal bijection using the Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            emd_loss += cost_matrix[row_ind, col_ind].sum() / N

        # Average over the batch
        emd_loss /= batch_size
        return emd_loss
import torch

class ApproximateEMD(torch.nn.Module):
    def __init__(self, num_samples=3000):
        """
        Args:
            num_samples: Number of points to sample for approximation.
        """
        super(ApproximateEMD, self).__init__()
        self.num_samples = num_samples

    def forward(self, S1, S2):
        """
        Args:
            S1: Tensor of shape (batch_size, N, 3) - first point set
            S2: Tensor of shape (batch_size, N, 3) - second point set (same size as S1)
        
        Returns:
            Approximate EMD loss value
        """
        batch_size, N, _ = S1.size()
        emd_loss = 0.0

        for i in range(batch_size):
            # Randomly sample points for approximation
            indices = torch.randperm(N)[:self.num_samples]
            sampled_S1 = S1[i, indices]  # shape: (num_samples, 3)
            sampled_S2 = S2[i, indices]  # shape: (num_samples, 3)

            # Compute pairwise distances between sampled points
            cost_matrix = torch.cdist(sampled_S1, sampled_S2, p=2)  # shape: (num_samples, num_samples)

            # Greedy algorithm without modifying the cost matrix
            loss = 0.0
            matched_rows = set()
            matched_cols = set()

            for _ in range(self.num_samples):
                # Mask out already matched points
                row_mask = torch.tensor([r not in matched_rows for r in range(self.num_samples)], device=cost_matrix.device)
                col_mask = torch.tensor([c not in matched_cols for c in range(self.num_samples)], device=cost_matrix.device)
                
                masked_cost_matrix = cost_matrix.clone()
                masked_cost_matrix[~row_mask[:, None]] = float('inf')
                masked_cost_matrix[:, ~col_mask] = float('inf')

                # Find the minimum value in the masked matrix
                min_val, min_idx = masked_cost_matrix.min(dim=1)
                min_point_idx = min_val.argmin().item()
                target_idx = min_idx[min_point_idx].item()

                # Accumulate the loss with the minimum distance
                loss += cost_matrix[min_point_idx, target_idx]

                # Mark these points as matched
                matched_rows.add(min_point_idx)
                matched_cols.add(target_idx)

            # Accumulate average loss over sampled points
            emd_loss += loss / self.num_samples

        # Average loss over batch
        emd_loss /= batch_size
        return emd_loss

class ApproximateEMDNew(torch.nn.Module):
    def __init__(self, num_samples=3000):
        """
        Args:
            num_samples: Number of points to sample for approximation.
        """
        super(ApproximateEMDNew, self).__init__()
        self.num_samples = num_samples

    def forward(self, S1, S2):
        """
        Args:
            S1: Tensor of shape (batch_size, N, 3) - first point set
            S2: Tensor of shape (batch_size, N, 3) - second point set (same size as S1)
        
        Returns:
            Approximate EMD loss value
        """
        batch_size, N, _ = S1.size()
        emd_loss = 0.0

        for i in range(batch_size):
            print(i)
            # Randomly sample points for approximation
            indices = torch.randperm(N)[:self.num_samples]
            sampled_S1 = S1[i, indices]  # shape: (num_samples, 3)
            sampled_S2 = S2[i, indices]  # shape: (num_samples, 3)

            # Compute pairwise distances between sampled points
            cost_matrix = torch.cdist(sampled_S1, sampled_S2, p=2)  # shape: (num_samples, num_samples)

            # Use a greedy algorithm to find approximate minimum matching
            loss = 0.0
            matched_S2 = torch.zeros_like(sampled_S1)

            # Iterative matching based on minimum distance
            for j in range(self.num_samples):
                min_val, min_idx = torch.min(cost_matrix, dim=1)
                min_point_idx = min_val.argmin().item()
                target_idx = min_idx[min_point_idx].item()

                # Add the minimum distance to the loss
                loss += cost_matrix[min_point_idx, target_idx]
                
                # Mark these points as matched (set distances to infinity to ignore in next iteration)
                cost_matrix[min_point_idx, :] = float('inf')
                cost_matrix[:, target_idx] = float('inf')
                
                # Store matched point (for completeness, not needed for loss calculation)
                matched_S2[min_point_idx] = sampled_S2[target_idx]

            # Accumulate average loss over sampled points
            emd_loss += loss / self.num_samples

        # Average loss over batch
        emd_loss /= batch_size
        return emd_loss
    
from chamferdist import ChamferDistance
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5,beta=0.5):
        super(CombinedLoss, self).__init__()
        self.chamfer = ChamferDistance()
        # self.emd = ApproximateEMDNew()
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

        ground_truth_scores = compute_confidence_score(pc3, pc2)  # Output shape: [32, 1000, 1]
        # print(ground_truth_scores.shape, pc1[3].shape)
        # print(ground_truth_scores, pc1[3])
        confidenseScoreLoss = self.mse(pc1[3],ground_truth_scores)

        seedGeneratorLoss = self.chamfer(pc1[1],pc2)

        cd = self.chamfer(pc1[0], pc2)
        upSamplingBlockLoss = self.alpha * cd #+ (1 - self.alpha) * emd_dist

        point2point=self.mse2(pc1[0],pc2)

        finalLoss = upSamplingBlockLoss + self.beta*seedGeneratorLoss+self.alpha*confidenseScoreLoss+point2point
        # finalLoss = self.alpha*confidenseScoreLoss
        return finalLoss
        # return self.alpha*confidenseScoreLoss, upSamplingBlockLoss, self.beta*seedGeneratorLoss






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

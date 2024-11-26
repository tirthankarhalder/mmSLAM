import torch
import torch.nn as nn
import numpy as np
# from pytorch3d.loss import emd_loss
from scipy.spatial import cKDTree
from pyemd import emd
from scipy.optimize import linear_sum_assignment
from chamferdist import ChamferDistance

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
    # x = x.detach().numpy()
    # y = y.detach().numpy()
    # diff = np.expand_dims(x, axis=1) - np.expand_dims(y, axis=0)
    # dist = np.linalg.norm(diff, axis=-1)
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    dist = torch.norm(diff, dim=-1)
    return dist
# Earth Mover's Distance (EMD) for one batch (source_points[0] and target_points[0])
def earth_movers_distance(source, target):
    print(source.shape,target.shape)
    # source = source.detach().numpy()
    # target = target.detach().numpy()
    print(source.shape,target.shape)
    print("inside emd")
    # Compute the cost matrix (pairwise distances between points)
    dist_matrix = compute_pairwise_distances(source, target)
    print("executed compute_pairwise_distances")
    # Uniform weights for source and target points
    
    source_weights = np.ones(source.shape[0]) / source.shape[0]
    target_weights = np.ones(target.shape[0]) / target.shape[0]
    dist_matrix = dist_matrix.detach().numpy()
    print(dist_matrix.shape)
    print(source_weights.shape,target_weights.shape)
    # Compute EMD
    emd_distance = emd(source_weights, target_weights, dist_matrix)

    return emd_distance
import torch
from pyemd import emd

def compute_emd_loss(batch_pcd1, batch_pcd2):
    """
    Args:
        batch_pcd1: Tensor of shape (batch_size, num_points, 3)
        batch_pcd2: Tensor of shape (batch_size, num_points, 3)
    
    Returns:
        emd_loss: Average EMD loss across the batch.
    """
    batch_size, num_points, _ = batch_pcd1.size()
    emd_losses = []

    for i in range(batch_size):
        print(i)
        # Extract point clouds for the current batch
        pcd1 = batch_pcd1[i]
        pcd2 = batch_pcd2[i]
        
        # Calculate pairwise distance matrix as the cost matrix
        cost_matrix = torch.cdist(pcd1, pcd2, p=2).cpu().detach().numpy()  # Shape: (num_points, num_points)
        
        # Uniform histograms for point masses (since we assume equal distribution)
        histogram1 = [1.0 / num_points] * num_points
        histogram2 = [1.0 / num_points] * num_points
        
        # Compute EMD using pyemd
        emd_value = emd(histogram1, histogram2, cost_matrix)
        emd_losses.append(emd_value)
        print(emd_value)
    
    # Average EMD loss across the batch
    emd_loss = sum(emd_losses) / batch_size
    return emd_loss

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

# class ApproximateEMD(torch.nn.Module):
#     def __init__(self, num_samples=3000):
#         """
#         Args:
#             num_samples: Number of points to sample for approximation.
#         """
#         super(ApproximateEMD, self).__init__()
#         self.num_samples = num_samples

#     def forward(self, S1, S2):
#         """
#         Args:
#             S1: Tensor of shape (batch_size, N, 3) - first point set
#             S2: Tensor of shape (batch_size, N, 3) - second point set (same size as S1)
        
#         Returns:
#             Approximate EMD loss value
#         """
#         batch_size, N, _ = S1.size()
#         emd_loss = 0.0

#         for i in range(batch_size):
#             # Randomly sample points for approximation
#             indices = torch.randperm(N)[:self.num_samples]
#             sampled_S1 = S1[i, indices]  # shape: (num_samples, 3)
#             sampled_S2 = S2[i, indices]  # shape: (num_samples, 3)

#             # Compute pairwise distances between sampled points
#             cost_matrix = torch.cdist(sampled_S1, sampled_S2, p=2)  # shape: (num_samples, num_samples)

#             # Greedy algorithm without modifying the cost matrix
#             loss = 0.0
#             matched_rows = set()
#             matched_cols = set()

#             for _ in range(self.num_samples):
#                 # Mask out already matched points
#                 row_mask = torch.tensor([r not in matched_rows for r in range(self.num_samples)], device=cost_matrix.device)
#                 col_mask = torch.tensor([c not in matched_cols for c in range(self.num_samples)], device=cost_matrix.device)
                
#                 masked_cost_matrix = cost_matrix.clone()
#                 masked_cost_matrix[~row_mask[:, None]] = float('inf')
#                 masked_cost_matrix[:, ~col_mask] = float('inf')

#                 # Find the minimum value in the masked matrix
#                 min_val, min_idx = masked_cost_matrix.min(dim=1)
#                 min_point_idx = min_val.argmin().item()
#                 target_idx = min_idx[min_point_idx].item()

#                 # Accumulate the loss with the minimum distance
#                 loss += cost_matrix[min_point_idx, target_idx]

#                 # Mark these points as matched
#                 matched_rows.add(min_point_idx)
#                 matched_cols.add(target_idx)

#             # Accumulate average loss over sampled points
#             emd_loss += loss / self.num_samples

#         # Average loss over batch
#         emd_loss /= batch_size
#         return emd_loss

# class ApproximateEMDNew(torch.nn.Module):
#     def __init__(self, num_samples=3000):
#         """
#         Args:
#             num_samples: Number of points to sample for approximation.
#         """
#         super(ApproximateEMDNew, self).__init__()
#         self.num_samples = num_samples

#     def forward(self, S1, S2):
#         """
#         Args:
#             S1: Tensor of shape (batch_size, N, 3) - first point set
#             S2: Tensor of shape (batch_size, N, 3) - second point set (same size as S1)
        
#         Returns:
#             Approximate EMD loss value
#         """
#         batch_size, N, _ = S1.size()
#         emd_loss = 0.0

#         for i in range(batch_size):
#             print(i)
#             indices = torch.randperm(N)[:self.num_samples]
#             sampled_S1 = S1[i, indices]  # shape: (num_samples, 3)
#             sampled_S2 = S2[i, indices]  # shape: (num_samples, 3)
#             cost_matrix = torch.cdist(sampled_S1, sampled_S2, p=2)  # shape: (num_samples, num_samples)
#             loss = 0.0
#             matched_S2 = torch.zeros_like(sampled_S1)
#             for j in range(self.num_samples):
#                 min_val, min_idx = torch.min(cost_matrix, dim=1)
#                 min_point_idx = min_val.argmin().item()
#                 target_idx = min_idx[min_point_idx].item()
#                 loss += cost_matrix[min_point_idx, target_idx]
#                 cost_matrix[min_point_idx, :] = float('inf')
#                 cost_matrix[:, target_idx] = float('inf')
#                 matched_S2[min_point_idx] = sampled_S2[target_idx]
#             emd_loss += loss / self.num_samples
#             print(emd_loss)
#         emd_loss /= batch_size
#         print(emd_loss)
#         return emd_loss
# import torch

# class ApproximateEMDNew1(torch.nn.Module):
#     def __init__(self, num_samples=3000):
#         """
#         Args:
#             num_samples: Number of points to sample for approximation.
#         """
#         super(ApproximateEMDNew1, self).__init__()
#         self.num_samples = num_samples

#     def forward(self, S1, S2):
#         """
#         Args:
#             S1: Tensor of shape (batch_size, N, 3) - first point set
#             S2: Tensor of shape (batch_size, N, 3) - second point set (same size as S1)
        
#         Returns:
#             Approximate EMD loss value
#         """
#         batch_size, N, _ = S1.size()
#         emd_loss = 0.0

#         for i in range(batch_size):
#             print(i)
#             indices = torch.randperm(N)[:self.num_samples]
#             sampled_S1 = S1[i, indices]  # shape: (num_samples, 3)
#             sampled_S2 = S2[i, indices]  # shape: (num_samples, 3)
#             cost_matrix = torch.cdist(sampled_S1, sampled_S2, p=2)  # shape: (num_samples, num_samples)
            
#             # Create a mask to keep track of used indices
#             row_mask = torch.ones(self.num_samples, dtype=torch.bool, device=S1.device)
#             col_mask = torch.ones(self.num_samples, dtype=torch.bool, device=S1.device)
#             loss = 0.0

#             for _ in range(self.num_samples):
#                 valid_cost_matrix = cost_matrix.clone()
#                 valid_cost_matrix[~row_mask, :] = float('inf')
#                 valid_cost_matrix[:, ~col_mask] = float('inf')
                
#                 min_val, min_row = torch.min(valid_cost_matrix, dim=0)
#                 min_val, min_col = torch.min(min_val, dim=0)
#                 min_row = min_row[min_col]
#                 loss += cost_matrix[min_row, min_col]
#                 row_mask[min_row] = False
#                 col_mask[min_col] = False
            
#             emd_loss += loss / self.num_samples
#             print(emd_loss)
#         emd_loss /= batch_size
#         return emd_loss


class ApproximatedEMDLoss(nn.Module):
    def __init__(self, epsilon=0.01, max_iter=50):
        """
        Initialize the approximated EMD loss class.

        Parameters:
        - epsilon: The approximation factor for (1 + epsilon).
        - max_iter: Maximum iterations for convergence in the Sinkhorn algorithm.
        """
        super(ApproximatedEMDLoss, self).__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter

    def forward(self, x, y):
        """
        Compute the (1 + epsilon) approximated EMD loss between two point clouds.

        Parameters:
        - x: Tensor of shape (batch_size, num_points, dim) representing the source point cloud.
        - y: Tensor of shape (batch_size, num_points, dim) representing the target point cloud.

        Returns:
        - loss: Approximated EMD loss.
        """
        batch_size, num_points, _ = x.size()
        
        # Compute pairwise cost matrix
        cost_matrix = torch.cdist(x, y, p=2)  # Shape: (batch_size, num_points, num_points)
        
        # Apply Sinkhorn-Knopp algorithm for optimal transport
        loss = self.sinkhorn_algorithm(cost_matrix, batch_size, num_points)
        
        return loss

    def sinkhorn_algorithm(self, cost_matrix, batch_size, num_points):
        """
        Compute the Sinkhorn distance for approximating EMD using (1 + epsilon).

        Parameters:
        - cost_matrix: Tensor of shape (batch_size, num_points, num_points).
        - batch_size: Number of batches.
        - num_points: Number of points in the point clouds.

        Returns:
        - Approximation of EMD loss.
        """
        # Uniform distributions for source and target
        r = torch.ones(batch_size, num_points, device=cost_matrix.device) / num_points
        c = torch.ones(batch_size, num_points, device=cost_matrix.device) / num_points
        
        # Regularization parameter
        epsilon = self.epsilon
        
        # Log-stabilized Sinkhorn iterations
        u = torch.zeros_like(r)
        v = torch.zeros_like(c)
        
        K = torch.exp(-cost_matrix / epsilon)  # Kernel matrix with regularization
        K_tilde = K / torch.sum(K, dim=1, keepdim=True)  # Normalize kernel
        
        for _ in range(self.max_iter):
            u = r / torch.sum(K_tilde * v.unsqueeze(1), dim=2)
            v = c / torch.sum(K_tilde * u.unsqueeze(2), dim=1)
        
        # Compute the Sinkhorn divergence (approximated transport cost)
        transport_plan = u.unsqueeze(2) * K * v.unsqueeze(1)
        loss = torch.sum(transport_plan * cost_matrix, dim=(1, 2))  # Batch-wise loss
        
        return loss.mean()  # Average over the batch

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5,beta=0.5):
        super(CombinedLoss, self).__init__()
        self.chamfer = ChamferDistance()
        self.emd = ApproximatedEMDLoss()
        # self.emd = EarthMoverDistance()
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
        # ground_truth_scores = compute_confidence_score(pc3, pc2)  # Output shape: [32, 1000, 1]
        # confidenseScoreLoss = self.mse(pc1[3],ground_truth_scores)
        # seedGeneratorLoss = self.chamfer(pc1[1],pc2)
        # cd = self.chamfer(pc1[0], pc2)
        # upSamplingBlockLossCHD = self.alpha*cd
        # upSamplingBlockLossEMD = self.emd(pc1[0],pc2)
        # finalLoss = upSamplingBlockLossCHD + self.beta*seedGeneratorLoss + self.alpha*confidenseScoreLoss +  upSamplingBlockLossEMD
        ########################################################

        # finalLoss = self.emd(pc1[0], pc2)#using custom class
        finalLoss = compute_emd_loss(pc1[0], pc2)#usign custom function
        return finalLoss

# if __name__=="__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = YourModel().to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     criterion = CombinedLoss(alpha=0.5).to(device)
#     num_epochs = 100
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

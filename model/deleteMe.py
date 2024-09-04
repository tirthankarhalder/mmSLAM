import torch
import torch.nn as nn
import torch.nn.functional as F

def farthest_point_sampling(xyz, npoint):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled point cloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    print("FPS: ", xyz.shape)
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids

class PointConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(PointConvLayer, self).__init__()
        self.conv1d = nn.Conv1d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1d(x)
        x = nn.ReLU(x)
        return x

class PointNetWithFPS(nn.Module):
    def __init__(self, input_dim, output_dim, num_samples):
        super(PointNetWithFPS, self).__init__()
        self.num_samples = num_samples
        self.conv1 = PointConvLayer(input_dim + 3, output_dim)  # F+3 -> output_dim

    def forward(self, xyz, features):
        """
        Input:
            xyz: point cloud coordinates, [B, N, 3]
            features: point cloud features, [B, N, F]
        Output:
            new_xyz: sampled point cloud coordinates, [B, num_samples, 3]
            new_features: point cloud features after convolution, [B, num_samples, output_dim]
        """
        # Step 1: Farthest Point Sampling
        B, N, _ = xyz.shape
        sampled_idx = farthest_point_sampling(xyz, self.num_samples)
        new_xyz = torch.gather(xyz, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, 3))

        # Step 2: Concatenate coordinates with features (F + 3)
        new_features = torch.gather(features, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, features.size(-1)))
        new_features = torch.cat([new_features, new_xyz], dim=-1)

        # Step 3: Apply point convolution
        new_features = new_features.permute(0, 2, 1)  # Change shape to [B, F+3, num_samples]
        new_features = self.conv1(new_features)
        new_features = new_features.permute(0, 2, 1)  # Back to [B, num_samples, output_dim]

        return new_xyz, new_features

# Example usage
B = 32  # Batch size
N = 1000  # Number of input points
F = 256  # Number of input features
num_samples = 500  # Number of points after sampling

# Generate random input data
xyz = torch.randn(B, N, 3)  # Point cloud coordinates [B, N, 3]
features = torch.randn(B, N, F)  # Point cloud features [B, N, F]

# Create the PointNet model with FPS and point convolution
pointnet_fps = PointNetWithFPS(input_dim=F, output_dim=256, num_samples=num_samples)

# Forward pass
new_xyz, new_features = pointnet_fps(xyz, features)

print(new_xyz.shape)  # Should be [32, 500, 3]
print(new_features.shape)  # Should be [32, 500, 256]


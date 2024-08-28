import torch
import torch.nn as nn

# Layer-1
class MLP(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MLP, self).__init__()
        
        # Define a single linear layer with ReLU activation
        self.fc = nn.Sequential(
            nn.Linear(input_channels, output_channels),  # Linear layer
            nn.ReLU()                                     # ReLU activation
        )

    def forward(self, x):
        return self.fc(x)


#LAyer-2
class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, k=20):
        batch_size, num_points, num_dims = x.size()
        x = x.permute(0, 2, 1)
        
        # Step 1: Find k-nearest neighbors
        dist = torch.cdist(x, x)
        _, idx = dist.topk(k, largest=False)

        # Step 2: Create the edge features
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x = x.permute(0, 2, 1)
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        edge_feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2)

        # Step 3: Apply the MLP
        return self.mlp(edge_feature)

class DGCNN(nn.Module):
    def __init__(self, input_channels, output_channels, k=20):
        super(DGCNN, self).__init__()
        self.k = k
        
        # Define the layers for the DGCNN
        self.conv1 = EdgeConv(input_channels, 64)
        self.conv2 = EdgeConv(64, 64)
        self.conv3 = EdgeConv(64, 128)
        self.conv4 = EdgeConv(128, 256)

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size, num_points, _ = x.size()

        # Apply EdgeConv layers
        x1 = self.conv1(x, self.k)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        x2 = self.conv2(x1, self.k)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x3 = self.conv3(x2, self.k)
        x3 = x3.max(dim=-1, keepdim=False)[0]

        x4 = self.conv4(x3, self.k)
        x4 = x4.max(dim=-1, keepdim=False)[0]

        # Concatenate features
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # Final 1D convolution
        x = self.conv5(x)

        return x

#Layer3
class MLP(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MLP, self).__init__()
        
        # Define the layers of the MLP
        self.fc = nn.Sequential(
            nn.Linear(input_channels, 128),  # First layer: 32 input channels -> 128 hidden units
            nn.ReLU(),                       # ReLU activation
            nn.Linear(128, output_channels)   # Second layer: 128 hidden units -> 64 output channels
        )

    def forward(self, x):
        return self.fc(x)


#layer-4
class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, k=20):
        batch_size, num_points, num_dims = x.size()
        x = x.permute(0, 2, 1)
        
        # Step 1: Find k-nearest neighbors
        dist = torch.cdist(x, x)
        _, idx = dist.topk(k, largest=False)

        # Step 2: Create the edge features
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x = x.permute(0, 2, 1)
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        edge_feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2)

        # Step 3: Apply the MLP
        return self.mlp(edge_feature)

class DGCNN(nn.Module):
    def __init__(self, input_channels, output_channels, k=20):
        super(DGCNN, self).__init__()
        self.k = k
        
        # Define the layers for the DGCNN
        self.conv1 = EdgeConv(input_channels, 64)
        self.conv2 = EdgeConv(64, 64)
        self.conv3 = EdgeConv(64, 128)
        self.conv4 = EdgeConv(128, 256)

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size, num_points, _ = x.size()

        # Apply EdgeConv layers
        x1 = self.conv1(x, self.k)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        x2 = self.conv2(x1, self.k)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x3 = self.conv3(x2, self.k)
        x3 = x3.max(dim=-1, keepdim=False)[0]

        x4 = self.conv4(x3, self.k)
        x4 = x4.max(dim=-1, keepdim=False)[0]

        # Concatenate features
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # Final 1D convolution
        x = self.conv5(x)

        return x

#layer -5 

class MaxPoolingModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MaxPoolingModel, self).__init__()
        
        # Define the convolutional layer
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=1)
        
        # Define the max pooling layer
        self.maxpool = nn.MaxPool1d(kernel_size=1)

    def forward(self, x):
        # x shape: [batch_size, input_channels, num_points]
        x = self.conv(x)  # Convolutional layer
        x = self.maxpool(x)  # Max pooling layer
        
        return x
    

    
# Parameters
N = 1000  # Number of points (can be any value you want)
input_channels = 3
output_channels = 32

# Create the model
model = MLP(input_channels, output_channels)

# Example input: N points with 3 input channels
x = torch.randn(N, input_channels)  # Shape: [N, 3]

# Forward pass
output = model(x)  # Shape: [N, 32]

# Print the output shape to verify
print("Output shape:", output.shape)  # Should be [N, 32]

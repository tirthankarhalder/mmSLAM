import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

class MLP(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_channels, output_channels),  
            nn.ReLU()                                    
        )

    def forward(self, x):
        return self.fc(x)


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

class ConfidenceScorePreditor(nn.Module):
    def __init__(self, mlp, dgCNN, maxPool, input_channels, output_channels):
        super(ConfidenceScorePreditor,self).__init__()

        self.mlp = mlp
        self.dgCNN = dgCNN
        self.maxPool = maxPool
    def forward(self, x):
        layer1 = self.mlp(x,3,32)
        layer2 = self.mlp(x,32,64)
        layer3 = self.dgCNN(x,64,64)
        layer4 = self.mlp(x,64,128)
        layer5 = self.mlp(x,128,256)
        layer6 = self.dgCNN(x,256,256)
        layer7 = self.maxPool(x,256,256)
        layer8 = self.mlp(x,)
        output = self.mlp(x,256,1)
        return output

if __name__ == "__main__":

        
    # Parameters
    N = 1000  # Number of points (can be any value you want)
    input_channels = 3
    output_channels = 32

    modelMLP = MLP(input_channels, output_channels)
    modelDGCNN = DGCNN(input_channels, output_channels)
    maxpool = MaxPoolingModel(input_channels,output_channels)
    confidencescorepredictor = ConfidenceScorePreditor(modelMLP,modelDGCNN,maxpool)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01)


    # Example input: N points with 3 input channels
    x = torch.randn(N, input_channels)  # Shape: [N, 3]

    # Forward pass
    output = confidencescorepredictor(x)  # Shape: [N, 32]


    # for epoch in range(100):  
    #     inputs = torch.randn(10)  
    #     labels = torch.tensor([1]) 

    #     optimizer.zero_grad()   

    #     outputs = model(inputs)  
    #     loss = criterion(outputs.unsqueeze(0), labels)  
    #     loss.backward()         
    #     optimizer.step()        

    #     if epoch % 10 == 0:
    #         print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')

    # with torch.no_grad():  
    #     test_input = torch.randn(10)
    #     test_output = model(test_input)
    #     predicted_class = torch.argmax(test_output)
    #     print(f'Predicted Class: {predicted_class.item()}')

    # torch.save(model.state_dict(), 'model.pth')

    print("Output shape:", output.shape)  # Should be [N, 32]

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

class DGCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super(DGCNNLayer, self).__init__()
        self.k = k
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        batch_size,num_points, num_dims = x.size()
        
        # Compute pairwise distance
        # x = x.permute(0, 2, 1)
        print("DGCNN before kNN: ",x.shape)
        idx = self.knn(x)
        print("DGCNN after kNN: ",x.shape)

        x = x.permute(0, 2, 1)
        
        # Dynamic graph construction
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, self.k, 1)
        
        # Concatenate the features
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
        
        # Apply convolution and batch normalization
        feature = F.relu(self.bn(self.conv(feature)))
        feature = feature.max(dim=-1, keepdim=False)[0]
        feature = feature.permute(0,2,1)
        print("DGCNN feature: ", feature.shape)

        return feature
    
    def knn(self, x):
        batch_size = x.size(0)
        num_points = x.size(1)
        
        inner = -2 * torch.matmul(x, x.transpose(2, 1))
        xx = torch.sum(x ** 2, dim=2, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        
        idx = pairwise_distance.topk(k=self.k, dim=-1)[1]
        return idx

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
        print("Edge conv x.shape: ", x.shape)
        x = x.permute(0, 2, 1)
        print("Edge conv x.shape: ", x.shape)
        dist = torch.cdist(x, x)
        _, idx = dist.topk(k, largest=False) #k neighbour

        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x = x.permute(0, 2, 1)
        print("EdgeConv processed x: ",x.shape)
        feature = x.view(batch_size * num_points, -1)[idx, :]
        print("Edgeconv feature:", feature.shape)
        feature = feature.view(batch_size, k, num_dims, num_dims)
        print("Edgeconv feature:", feature.shape)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        edge_feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2)

        return self.mlp(edge_feature)

class DGCNN(nn.Module):
    def __init__(self, input_channels, output_channels, k=20):
        super(DGCNN, self).__init__()
        self.k = k
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
        print("DGCNN x: ", x.shape)
        x1 = self.conv1(x, self.k)
        print("DGCNN x1: ", x1.shape)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        x2 = self.conv2(x1, self.k)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x3 = self.conv3(x2, self.k)
        x3 = x3.max(dim=-1, keepdim=False)[0]

        x4 = self.conv4(x3, self.k)
        x4 = x4.max(dim=-1, keepdim=False)[0]

        # Concatenate 
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # 1D convolution
        x = self.conv5(x)

        return x

class MaxPooling(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MaxPooling, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)

    def forward(self, x):
        # x shape: [batch_size, input_channels, num_points]
        x = x.permute(0,2,1)
        print("Maxpool: ",x.shape)
        x = self.conv(x)  
        x = self.maxpool(x) 
        x = x.permute(0,2,1)
        return x

class ConfidenceScorePredictor(nn.Module):
    def __init__(self):
        super(ConfidenceScorePredictor,self).__init__()

        self.mlp1 = MLP(3,32)
        self.mlp2 = MLP(32,64)
        self.mlp3 = MLP(64,128)
        self.mlp4 = MLP(128,256)
        self.mlp5 = MLP(576,256)
        self.mlp6 = MLP(256,1)
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_channels, output_channels),  
        #     nn.ReLU()                                    
        # )
        self.dgCNN1 = DGCNNLayer(64,64)
        self.dgCNN2 = DGCNNLayer(256,256)
        # self.dgCNN1 = DGCNN(64,64)
        # self.dgCNN2 = DGCNN(256,256)
        self.maxPool = MaxPooling(256,256)

    def forward(self, x):
        layer1 = self.mlp1(x)
        print("layer1.shape",layer1.shape)
        layer2 = self.mlp2(layer1)
        print("layer2.shape: ", layer2.shape)
        layer3 = self.dgCNN1(layer2)
        print("layer3.shape: ", layer3.shape)
        layer4 = self.mlp3(layer3)
        print("layer4.shape: ",layer4.shape)
        layer5 = self.mlp4(layer4)
        print("layer5.shape: ",layer5.shape)
        layer6 = self.dgCNN2(layer5)
        print("layer6.shape: ",layer6.shape)
        layer7 = self.maxPool(layer6)
        print("layer7.shape: ",layer7.shape)
        layer8 = torch.cat((N*layer7,layer6,layer3),dim=1)
        print("layer8.shape: ",layer8.shape)
        layer9 = self.mlp5(layer8)
        print("layer9.shape: ",layer9.shape)
        output = self.mlp6(layer9)
        return output

if __name__ == "__main__":
    batch_size = 32
    N = 1000  
    input_channels = 3
    output_channels = 32
    device = torch.device("cpu")
    # device = torch.device("cuda:0")
    # base.check_backend_library("cupy")

    model = ConfidenceScorePredictor().to(device)
    x = torch.randn(batch_size,N, 3)
    output = model(x)
    print("Output shape:", output.shape)


    # modelMLP = MLP(input_channels, output_channels)
    # modelDGCNN = DGCNN(input_channels, output_channels)
    # maxpool = MaxPoolingModel(input_channels,output_channels)
    # confidencescorepredictor = ConfidenceScorePreditor(modelMLP,modelDGCNN,maxpool)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01)


    # Example input: N points with 3 input channels
    # x = torch.randn(N, input_channels)  # Shape: [N, 3]

    # Forward pass
    # output = ConfidenceScorePreditor(x)  # Shape: [N, 32]


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

    # print("Output shape:", output.shape)  # Should be [N, 32]

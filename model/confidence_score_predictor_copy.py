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

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, k, emb_dims, dropout, output_channels=40):
        super(DGCNN, self).__init__()
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        print("DGCNN: ",x.shape)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = x.permute(0,2,1,3)
        print('graph x: ',x.shape)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        print("conv1:",x1.shape)
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        print("con5 :",x.shape)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class MaxPooling(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MaxPooling, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=1)

    def forward(self, x):
        # x shape: [batch_size, input_channels, num_points]
        x = self.conv(x)  
        x = self.maxpool(x) 
        
        return x

class ConfidenceScorePredictor(nn.Module):
    def __init__(self):
        super(ConfidenceScorePredictor,self).__init__()

        self.mlp1 = MLP(3,32)
        self.mlp2 = MLP(32,64)
        self.mlp3 = MLP(64,128)
        self.mlp4 = MLP(128,256)
        self.mlp5 = MLP(512+64,256)
        self.mlp6 = MLP(256,1)
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_channels, output_channels),  
        #     nn.ReLU()                                    
        # )
        self.dgCNN1 = DGCNN(k=20,emb_dims=1024,dropout=0.5,output_channels=64)
        self.dgCNN2 = DGCNN(k=20,emb_dims=1024,dropout=0.5,output_channels=256)
        self.maxPool = MaxPooling(256,256)

    def forward(self, x):
        layer1 = self.mlp1(x)
        print("layer1.shape: ", layer1.shape)
        layer2 = self.mlp2(layer1)
        print("layer2.shape: ", layer2.shape)
        layer3 = self.dgCNN1(layer2)
        print("layer3.shape: ", layer3.shape)
        layer4 = self.mlp3(layer3)
        print("layer4.shape: ",layer4.shape)
        layer5 = self.mlp4(layer4)
        print("layer5 ", layer5.shape)
        layer6 = self.dgCNN2(layer5)
        layer7 = self.maxPool(layer6)
        layer8 = self.mlp5(layer7)
        layer = torch.cat((N*layer7,layer3,layer6),dim=1)
        output = self.mlp6(x)
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

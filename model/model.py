import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
class MLP(nn.Module):
    def __init__(self, input_channels, output_channels,num_samples = None,relu=True,activation=True,fps=False):
        super(MLP, self).__init__()
        self.num_samples = num_samples
        if activation:
            if relu:
                self.fc = nn.Sequential(
                    nn.Linear(input_channels, output_channels),  
                    nn.ReLU()                                    
                )
            else:
                self.fc = nn.Sequential(
                    nn.Linear(input_channels, output_channels),  
                    nn.Sigmoid()                                    
                )

        else:
            self.fc = nn.Sequential(nn.Linear(input_channels, output_channels))

    def forward(self,x):
        
        output = self.fc(x)
        return output


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
        # print("DGCNN before kNN: ",x.shape)
        idx = self.knn(x)
        # print("DGCNN after kNN: ",x.shape)

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
        # print("DGCNN feature: ", feature.shape)

        return feature
    
    def knn(self, x):
        batch_size = x.size(0)
        num_points = x.size(1)
        
        inner = -2 * torch.matmul(x, x.transpose(2, 1))
        xx = torch.sum(x ** 2, dim=2, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        
        idx = pairwise_distance.topk(k=self.k, dim=-1)[1]
        return idx


class MaxPooling(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MaxPooling, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=1000)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.maxpool(x) 
        x = x.squeeze(-1)
        return x

class ConfidenceScorePredictor(nn.Module):
    def __init__(self):
        super(ConfidenceScorePredictor,self).__init__()

        self.mlp1 = MLP(3,32)
        self.mlp2 = MLP(32,64)
        self.mlp3 = MLP(64,128,num_samples=500)
        self.mlp4 = MLP(128,256)
        self.mlp5 = MLP(576,256)
        self.mlp6 = MLP(256,1,relu=False)
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_channels, output_channels),  
        #     nn.ReLU()                                    
        # )
        self.dgCNN1 = DGCNNLayer(64,64)
        self.dgCNN2 = DGCNNLayer(256,256)
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
        layer_repeat = layer7.unsqueeze(1).repeat(1,1000,1)
        layer8 = torch.cat((layer_repeat,layer6,layer3),dim=2)
        print("layer8.shape: ",layer8.shape)
        layer9 = self.mlp5(layer8)
        print("layer9.shape: ",layer9.shape)
        output = self.mlp6(layer9)
        return output
    
class PointConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels,stride=2):
        super(PointConvLayer, self).__init__()
        self.conv1d = nn.Conv1d(input_channels , output_channels, kernel_size=1,stride=stride)

    def forward(self, x):
        x = x.permute(0,2,1)
        print("PointConv X.shape", x.shape)
        x = self.conv1d(x)
        # x = F.relu(x)  
        x = x.permute(0,2,1)
        return x
    
class MaxPooling(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MaxPooling, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=1000)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.maxpool(x) 
        x = x.squeeze(-1)
        return x
    
def farthest_point_sampling(xyz, npoint):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
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
   
class NoideAwareFeatureExtractor(nn.Module):
    def __init__(self):
        super(NoideAwareFeatureExtractor,self).__init__()
        self.mlp1 = MLP(4,64)
        self.mlp2 = MLP(64,256)
        self.mlp3 = MLP(259,256)
        self.mlp4 = MLP(256,384)
        self.mlp5 = MLP(387,384)
        self.mlp6 = MLP(384,384)

        self.pConv1 = PointConvLayer(256,256)
        self.pConv2 = PointConvLayer(256,256)

        self.maxPool = MaxPooling(512,512)




    def forward(self,model_weights,x):
        layer1 = torch.cat((model_weights,x),dim=2)
        print("layer1.shape",layer1.shape)
        layer2 = self.mlp1(layer1)
        print("layer2.shape: ", layer2.shape)
        layer3 = self.mlp2(layer2)
        print("layer3.shape: ", layer3.shape)
        layer4 = self.pConv1(layer3)
        print("layer4.shape: ",layer4.shape)
        concat_layer = torch.cat((layer4,x),dim=2)
        print("concat_layer.shape: ",concat_layer.shape)
        layer5 = self.mlp3(concat_layer)
        print("layer5.shape: ",layer5.shape)
        layer6 = self.mlp4(layer5)
        print("layer6.shape: ",layer6.shape)
        layer7 = self.pConv2(layer6)
        print("layer7.shape: ",layer7.shape)
        # layer_repeat = layer7.unsqueeze(1).repeat(1,1000,1)
        # layer8 = torch.cat((layer_repeat,layer6,layer3),dim=2)
        layer8 = self.mlp5(layer7)
        print("layer8.shape: ",layer8.shape)
        layer9 = self.mlp6(layer8)
        print("layer9.shape: ",layer9.shape)
        output = self.maxPools(layer9)
        return output

if __name__ == "__main__":
    batch_size = 32
    N = 1000  
    input_channels = 3
    output_channels = 32
    device = torch.device("cpu")

    score = ConfidenceScorePredictor().to(device)
    x = torch.randn(batch_size,N, 3)
    model1 = score(x)
    print("Model1 Output shape:", model1.shape)
    print("==============================")

    noise = NoideAwareFeatureExtractor().to(device)

    model2 = noise(model1,x)

    print("Model2 Output shape:", model2.shape)


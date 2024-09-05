import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from confidenceScorePredictor import ConfidenceScorePredictor



def farthest_point_sampling(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    print("FPS: ",xyz.shape)
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

# class MLP(nn.Module):
#     def __init__(self, input_channels, output_channels,relu=True,activation=True):
#         super(MLP, self).__init__()
#         if activation:
#             if relu:
#                 self.fc = nn.Sequential(
#                     nn.Linear(input_channels, output_channels),  
#                     nn.ReLU()                                    
#                 )
#             else:
#                 self.fc = nn.Sequential(
#                     nn.Linear(input_channels, output_channels),  
#                     nn.Sigmoid()                                    
#                 )

#         else:
#             self.fc = nn.Sequential(nn.Linear(input_channels, output_channels))

#     def forward(self, x):
#         return self.fc(x)
class MLP(nn.Module):
    def __init__(self, input_channels, output_channels,relu=True,activation=True):
        super(MLP, self).__init__()
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

    def forward(self, x):
        return self.fc(x)

class PointConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(PointConvLayer, self).__init__()
        self.conv1d = nn.Conv1d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = x.permute(0,2,1)
        print("PointConv X.shape", x.shape)
        x = self.conv1d(x)
        # x = F.relu(x)  
        x = x.permute(0,2,1)
        return x

# class PointConvLayer(nn.Module):
#     def __init__(self, input_channels, output_channels,num_samples):
#         super(PointConvLayer, self).__init__()
#         self.num_samples = num_samples
#         self.conv1d = nn.Conv1d(input_channels, output_channels, kernel_size=1)

#     def forward(self, x, features):
#         B, N, _ = x.shape
#         sampled_idx = farthest_point_sampling(x, self.num_samples)
#         new_xyz = torch.gather(x, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, 3))

#         new_features = torch.gather(features, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, features.size(-1)))
#         new_features = torch.cat([new_features, new_xyz], dim=-1)
#         print("PointConv new_features.shape", new_features.shape)

#         new_features = new_features.permute(0, 2, 1)  # Change shape to [B, F+3, num_samples]
#         new_features = self.conv1d(new_features)
#         new_features = new_features.permute(0, 2, 1)
#         # x = x.permute(0,2,1)
#         print("PointConv new_xyz.shape", new_xyz.shape)
#         print("PointConv new_features.shape", new_features.shape)

#         return new_xyz,new_features

class MaxPooling(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MaxPooling, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=1000)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.maxpool(x) 
        x = x.squeeze(-1)
        return x
    



class NoideAwareFeatureExtractor(nn.Module):
    def __init__(self):
        super(NoideAwareFeatureExtractor,self).__init__()
        self.mlp1 = MLP(4,64)
        self.mlp2 = MLP(64,256)
        self.mlp3 = MLP(259,256)
        self.mlp4 = MLP(256,384)
        self.mlp5 = MLP(387,384)
        self.mlp6 = MLP(384,512)

        self.pConv1 = PointConvLayer(256,256)
        self.pConv2 = PointConvLayer(384,384)

        self.maxPool = MaxPooling(512,512)




    def forward(self,model_weights,x):
        # batch,num_points,_ = x.size()
        confidenseScore = ConfidenceScorePredictor().to(device)
        confidenseScoreWeights = confidenseScore(x)
        print("==============================")
        
        feature_concat = torch.cat((confidenseScoreWeights,x),dim=2)
        print("Concat layer.shape",feature_concat.shape)
        layer1 = self.mlp1(feature_concat)
        print("layer1.shape: ", layer1.shape)
        layer2 = self.mlp2(layer1)
        print("layer2.shape: ", layer2.shape)
        # features_dim = torch.randn(batch_size, num_points, 256)
        layer3 = self.pConv1(layer2)
        print("layer3.shape: ",layer3.shape)


        B, N, _ = x.shape
        sampled_idx = farthest_point_sampling(x, 500)
        new_xyz = torch.gather(x, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, 3))

        new_features = torch.gather(layer3, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, layer3.size(-1)))
        new_features = torch.cat([new_features, new_xyz], dim=-1)
        print("new_features.shape", new_features.shape)
        print("new_xyz.shape", new_xyz.shape)


        layer4 = self.mlp3(new_features)

        print("layer4.shape: ",layer4.shape)
        layer5 = self.mlp4(layer4)
        print("layer5.shape: ",layer5.shape)
        layer6 = self.pConv2(layer5)
        print("layer6.shape: ",layer6.shape)

        B, N, _ = x.shape
        sampled_idx = farthest_point_sampling(x, 1000)
        new_xyz = torch.gather(x, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, 3))

        new_features = torch.gather(layer6, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, layer6.size(-1)))
        new_features = torch.cat([new_features, new_xyz], dim=-1)
        print("new_features.shape", new_features.shape)

        layer7 = self.mlp5(new_features)
        print("new_xyz.shape", new_xyz.shape)

        print("layer7.shape: ",layer7.shape)
        layer8 = self.mlp6(layer7)
        print("layer8.shape: ",layer8.shape)
        output = self.maxPool(layer8)
        print("output.shape: ",output.shape)
        return output

if __name__ == "__main__":
    batch_size = 32
    N = 1000  
    input_channels = 3
    output_channels = 32
    device = torch.device("cpu")
    # score = ConfidenceScorePredictor().to(device)
    noise = NoideAwareFeatureExtractor().to(device)

    x = torch.randn(batch_size,N, 3)
    model2 = noise(x)

    print("Model2 Output shape:", model2.shape)
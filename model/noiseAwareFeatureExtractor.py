import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from .confidenceScorePredictor import ConfidenceScorePredictor



def farthest_point_sampling(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    # print("FPS: ",xyz.shape)
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
    # print("centroi.shape: ", centroid.shape)
    return centroids

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
        # print("PointConv X.shape", x.shape)
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
        print("x.shape: ",x.shape)
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




    def forward(self,x):
        # batch,num_points,_ = x.size()
        confidenseScore = ConfidenceScorePredictor().to(x.device)
        confidenseScoreWeights = confidenseScore(x)
        print("==============================")
        print("confidenseScoreWeights.shape",confidenseScoreWeights.shape)
        feature_concat = torch.cat((confidenseScoreWeights,x),dim=2)
        print("Concat layer.shape",feature_concat.shape)
        layer1 = self.mlp1(feature_concat)
        print("layer1.shape: ", layer1.shape)
        layer2 = self.mlp2(layer1)
        print("layer2.shape: ", layer2.shape)
        # features_dim = torch.randn(batch_size, num_points, 256)
        #have to check
        # layer_down = F.interpolate(layer2.permute(0, 2, 1), size=(500), mode='linear', align_corners=False).permute(0, 2, 1)
        # print("layer_down.shape: ",layer_down.shape)

        B, N, _ = x.shape
        sampled_idx = farthest_point_sampling(x, 500)

        new_xyz = torch.gather(x, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, 3))
        new_features = torch.gather(layer2, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, layer2.size(-1)))
        # new_features = torch.cat([new_features, new_xyz], dim=-1)
        print("new_features.shape", new_features.shape)
        print("new_xyz.shape", new_xyz.shape)

        layer3 = self.pConv1(new_features)
        print("layer3.shape: ",layer3.shape)


        concat_layer = torch.cat((new_xyz,layer3),dim=2)
        print("concat_layer.shape: ", concat_layer.shape)

        layer4 = self.mlp3(concat_layer)
        print("layer4.shape: ",layer4.shape)

        layer5 = self.mlp4(layer4)
        print("layer5.shape: ",layer5.shape)

        B, N, _ = new_xyz.shape
        sampled_idx = farthest_point_sampling(new_xyz, 250)

        new_xyz = torch.gather(new_xyz, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, 3))
        new_features = torch.gather(layer5, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, layer5.size(-1)))
        # new_features = torch.cat([new_features, new_xyz], dim=-1)
        print("new_features.shape", new_features.shape)
        print("new_xyz.shape", new_xyz.shape)

        layer6 = self.pConv2(new_features)
        print("layer6.shape: ",layer6.shape)

        concat_layer = torch.cat((new_xyz,layer6),dim=2)
        print("concat_layer.shape: ", concat_layer.shape)

        layer7 = self.mlp5(concat_layer)
        print("layer7.shape: ",layer7.shape)

        B, N, _ = x.shape
        sampled_idx = farthest_point_sampling(new_xyz, 125)

        new_xyz = torch.gather(x, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, 3))
        new_features = torch.gather(layer7, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, layer7.size(-1)))
        # new_features = torch.cat([new_features, new_xyz], dim=-1)
        print("new_features.shape", new_features.shape)
        print("new_xyz.shape", new_xyz.shape)

        layer8 = self.mlp6(new_features)
        print("layer8.shape: ",layer8.shape)

        output=torch.max(layer8,dim=1).values
        print("output.shape: ",output.shape)

        return output,confidenseScoreWeights

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
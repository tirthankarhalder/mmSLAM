import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
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
        self.mlp3 = MLP(64,128)
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
        print("layer_repeat.shape: ",layer_repeat.shape)
        layer8 = torch.cat((layer_repeat,layer6,layer3),dim=2)
        print("layer8.shape: ",layer8.shape)
        layer9 = self.mlp5(layer8)
        print("layer9.shape: ",layer9.shape)
        output = self.mlp6(layer9)
        print("output.shape: ",output.shape)
        return output

if __name__ == "__main__":
    batch_size = 32
    N = 1000  
    input_channels = 3
    output_channels = 32
    device = torch.device("cpu")

    model = ConfidenceScorePredictor().to(device)
    x = torch.randn(batch_size,N, 3)
    output = model(x)
    print("Output shape:", output.shape)


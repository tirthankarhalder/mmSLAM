import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from noiseAwareFeatureExtractor import NoideAwareFeatureExtractor
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

class UpConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, output_size):
        super(UpConv1D, self).__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=output_size)
        
    def forward(self, x):
        x= x.permute(0,2,1)
        x = self.upconv(x)
        x= x.permute(0,2,1)
        return x
    

class SeedGenerator(nn.Module):
    def __init__(self):
        super(SeedGenerator,self).__init__()

        self.mlp1 = MLP(640,256)
        self.mlp2 = MLP(256,128)
        self.mlp3 = MLP(640,128)
        self.mlp4 = MLP(128,128)
        self.mlp5 = MLP(128,64)
        self.mlp6 = MLP(64,3,activation=False)
        self.upCon = UpConv1D(512,128,1)

    def forward(self, x):

        # noiseAwareFF = NoideAwareFeatureExtractor().to(device)
        # noiseAwareFFWeights = noiseAwareFF(x)
        layer1 = self.upCon(x)
        print("layer1.shape",layer1.shape)
        # feature_extractor = NoideAwareFeatureExtractor().to(device)
        feature_extractor = torch.randn(32,512)
        layer_repeat = feature_extractor.unsqueeze(1).repeat(1,1000,1)
        print("layer_repeat.shape: ", layer_repeat.shape)
        concat_layer1 = torch.cat(layer1,layer_repeat)
        print("concat_layer1.shape: ", concat_layer1.shape)
        layer2 = self.mlp1(concat_layer1)
        print("layer2.shape: ", layer2.shape)
        layer3 = self.mlp2(layer2)
        print("layer3.shape: ", layer3.shape)
        layer4 = self.mlp3(layer3)
        print("layer4.shape: ",layer4.shape)
        layer5 = self.mlp4(layer4)
        print("layer5.shape: ",layer5.shape)
        layer6 = self.mlp5(layer5)
        print("layer6.shape: ",layer6.shape)
        output = self.mlp6(layer6)
        print("output.shape: ",output.shape)
        return output

if __name__ == "__main__":
    batch_size = 32
    N = 1000  
    input_channels = 512
    output_channels = 128
    device = torch.device("cpu")

    model = SeedGenerator().to(device)
    x = torch.randn(batch_size,1, 512)
    output = model(x)
    print("Output shape:", output.shape)


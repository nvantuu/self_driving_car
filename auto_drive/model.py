import torch
import torch.nn as nn

class AutoPilotModel(nn.Module):

    def __init__(self):
        super(AutoPilotModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3)),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.Dropout(0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 2 * 33, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        # print("size: ", x.size())
        input = x.view(x.size(0), 3, 70, 320)
        output = self.conv_layers(input)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output

    def predict(self, img):
        img = torch.from_numpy(img).float()
        with torch.no_grad():
            angleSteering = self(img)

        return angleSteering




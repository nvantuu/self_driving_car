from auto_drive.data_module import *
from auto_drive.model import AutoPilotModel
from auto_drive.process_data import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train():
    df = read_df(c.log_path)
    list_samples = generate_samples(df)

    train_samples, val_samples = data_split(list_samples)

    training_set = SelfDrivingCarData(train_samples)
    train_dataloader = DataLoader(training_set, **c.loader_params)

    validation_set = SelfDrivingCarData(val_samples)
    val_dataloader = DataLoader(validation_set, **c.loader_params)

    learning_rate = 0.0001
    model = AutoPilotModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is: ', device)

    def toDevice(datas, device):
        imgs, angles = datas
        return imgs.float().to(device), angles.float().to(device)


    for epoch in range(c.NUM_EPOCHS):
        print("########################################################")
        print("                    EPOCH: ", epoch, "                    ")
        print("########################################################")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training
        train_loss = 0
        model.train()
        for batch_number, data in enumerate(train_dataloader):

            # print("Training Batch Number: " , batch_number)
            # Transfer to GPU
            data = toDevice(data, device)

            # Model computations
            optimizer.zero_grad()
            imgs, angles = data
            # print("training image: ", imgs.shape)
            outputs = model(imgs)
            loss = criterion(outputs, angles.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_number % 100 == 0:
                print('Train Loss: %.3f '
                      % (train_loss / ((batch_number + 1) * 3)))

        # Validation
        model.eval()
        valid_loss = 0
        with torch.set_grad_enabled(False):
            for batch_number, data in enumerate(val_dataloader):
                # print("Validing Batch Number: " , batch_number)
                # Transfer to GPU
                data = toDevice(data, device)

                # Model computations
                optimizer.zero_grad()

                imgs, angles = data
                # print("Validation image: ", imgs.shape)
                outputs = model(imgs)
                loss = criterion(outputs, angles.unsqueeze(1))

                valid_loss += loss.item()

                avg_valid_loss = valid_loss / (batch_number + 1)
                if batch_number % 100 == 0:
                    print('Valid Loss: %.3f\n'
                          % (valid_loss / (batch_number + 1)))
        learning_rate = learning_rate * 0.7

    path_save = os.path.join(c.save_path, 'model_22_epoch.pth')
    torch.save(model.state_dict(), path_save)

if __name__ == '__main__':
    print(1e-4)
    train()
    # path_save = os.path.join(c.save_path, 'model_1_epoch.pth')
    # model = AutoPilotModel()
    # model.load_state_dict(torch.load(path_save))





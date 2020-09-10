import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from unet_model import Unet
from dataset import LiverDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1

x_transforms = transforms.Compose([
    transforms.ToTensor(),  # -> [0,1]
    transforms.Normalize([0.5], [0.5])  # ->[-1,1]
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()



def train_model(model, criterion, optimizer, dataload, num_epochs=2):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model


def train():
    model = Unet(1, 1).to(device)
    batch_size = BATCH_SIZE
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("./data/membrane/train/image","./data/membrane/train/label",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_model(model, criterion, optimizer, dataloaders)


def test():
    model = Unet(1, 1)
    #model.load_state_dict(torch.load(args.ckp,map_location='cpu'))
    liver_dataset = LiverDataset("./data/membrane/test/image","./data/membrane/test/predict", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x, _ in dataloaders:
            y=model(x)
            img_y=torch.squeeze(y).numpy()
            plt.imshow(img_y)
            plt.pause(0.01)



train()
test()
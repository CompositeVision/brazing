import torch
import torch.nn as nn
import torch.optim
import torchvision
import torchvision.transforms.v2 as T
import tqdm
from torch.utils.data import DataLoader

from dataset import Dataset

torchvision.disable_beta_transforms_warning()

model = nn.Sequential(
    nn.Conv2d(1, 128, (3,3), padding=(1,1)),
    nn.ReLU(),
    nn.Conv2d(128, 128, (3,3), padding=(1,1)),
    nn.ReLU(),
    nn.Conv2d(128, 128, (3,3), padding=(1,1)),
    nn.ReLU(),
    nn.Conv2d(128, 1, (3,3), padding=(1,1)),
)


transforms = T.Compose([
    T.ToImageTensor(),
    T.ConvertDtype(),
    T.RandomAffine(180, (0.3, 0.3), shear=0.5, fill=167),
    T.Normalize(mean=[167], std=[44])
])
ds = Dataset(transforms)

train_loader = DataLoader(ds, batch_size=8, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train_epoch(model: torch.nn.Module, train_loader: DataLoader):
    model.train()
    losses = []
    for batch in tqdm.tqdm(train_loader):
        x, y_true = batch
        model.zero_grad()
        y_pred = model(x)
        loss = criterion(input=y_pred, target=y_true.float())
        loss.backward()

        optimizer.step()

        losses.append(loss.item())
    
    print(sum(losses)/len(losses))

# Train
for i in range(5):
    train_epoch(model, train_loader)

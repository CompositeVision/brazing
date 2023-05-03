import torch
import torch.nn as nn
import torch.optim
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T
import tqdm
import unet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import Dataset, get_image_and_mask_files


class Model(nn.Module):
    def __init__(self, extractor, n_filters) -> None:
        super().__init__()
        self.extractor = extractor
        self.classifier = nn.Conv2d(n_filters + 1, 1, 3, 1, 1, bias=True)
    def forward(self, x):
        features = self.extractor(x)
        last = torch.concat((x, features), dim=1)
        return self.classifier(last)


def get_model(n_filters=32):
    return unet.UNet(1, 1, num_encoding_blocks=3, normalization="batch", dropout=0.2, padding=1)
    extractor = nn.Sequential(
        nn.Conv2d(1, n_filters, 3, 1, 1, bias=False),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),
        nn.Conv2d(n_filters, n_filters, 3, 1, 1, bias=False),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),
        nn.Conv2d(n_filters, n_filters, 3, 1, 1, bias=False),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),
        nn.Conv2d(n_filters, n_filters, 3, 1, 1, bias=False),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),
        nn.Conv2d(n_filters, n_filters, 3, 1, 1, bias=False),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),
        nn.Conv2d(n_filters, n_filters, 3, 1, 1, bias=False),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),
        nn.Conv2d(n_filters, n_filters, 3, 1, 1, bias=False),
        nn.BatchNorm2d(n_filters),
        nn.ReLU(),
    )
            
    return Model(extractor, n_filters)

def get_transforms_train():
    transforms = T.Compose([
        T.ToImageTensor(),
        T.ConvertDtype(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomAffine(45, shear=20),
        # T.Normalize(mean=[167], std=[44])
    ])
    return transforms

def get_transforms_val():
    transforms = T.Compose([
        T.ToImageTensor(),
        T.ConvertDtype(),
        # T.Normalize(mean=[167], std=[44])
    ])
    return transforms

def train_val_epoch(
    model: torch.nn.Module, 
    criterion,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None
):
    model.train()
    train_losses = []
    for batch in tqdm.tqdm(train_loader, desc="train"):
        x, y_true = batch
        y_pred = model(x.cuda())
        loss = criterion(input=y_pred, target=y_true.float().cuda())
        loss.backward()

        optimizer.step()

        model.zero_grad()

        train_losses.append(loss.item())
    
    if val_loader is None:
        return train_losses, []
    
    model.eval()
    with torch.no_grad():
        val_losses = []
        for batch in tqdm.tqdm(val_loader, desc="val"):
            x, y_true = batch
            y_pred = model(x.cuda())
            loss = criterion(input=y_pred, target=y_true.float().cuda())
            val_losses.append(loss.item())

    if scheduler is not None:
        scheduler.step(sum(val_losses)/len(val_losses))

    return train_losses, val_losses


if __name__ == "__main__":
    image_files_train, mask_files_train, image_files_val, mask_files_val = get_image_and_mask_files("processed", 0.1, random_seed=1)
    print(f"train/val: {len(image_files_train)}/{len(image_files_val)}")
    train_ds = Dataset(image_files_train, mask_files_train, get_transforms_train())
    val_ds = Dataset(image_files_val, mask_files_val, get_transforms_val())

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = get_model(n_filters=32).cuda()

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.2], device="cuda"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=5, verbose=True)

    train_loss_epoch = []
    val_loss_epoch = []
    val_best = 999.0
    for i in range(100):
        train_losses, val_losses = train_val_epoch(model, criterion, optimizer, train_loader, val_loader, scheduler)
        def avg(lst):
            return sum(lst)/len(lst) if lst else float("nan")
        train_loss_epoch.append(avg(train_losses))
        val_loss_epoch.append(avg(val_losses))
        plt.figure()
        plt.plot(train_loss_epoch, label="train")
        plt.plot(val_loss_epoch, label="val")
        plt.legend()
        plt.savefig("metrics.png")
        plt.close()
        print(f"Epoch {i}: {avg(train_losses):.5f}/{avg(val_losses):.5f}")
    
        if val_loss_epoch[-1] < val_best:
            val_best = val_loss_epoch[-1]
            model.cpu()
            with open("model.bin", "wb") as f:
                torch.save(model.state_dict(), f)
            model.cuda()
    

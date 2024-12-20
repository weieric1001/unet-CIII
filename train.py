import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pathlib import Path

from unet import UNet
from utils.dataset import CIIDataset
from utils.logger import set_logger

pre_folder = "./CIII/pre/front/processed"
post_folder = "./CIII/post/front/processed"
dir_checkpoint = "./checkpoints/"


def train(
    model,
    device,
    epochs: int,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_size: float = 128,
    amp: bool = False,
    weight_decay: float = 1e-8,
):
    logger = set_logger()
    # 1. Create dataset
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = CIIDataset(pre_folder, post_folder, transform=transform)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    # 3. Create data loaders
    loader_args = dict(
        batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logger.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images Size:     {img_size}
        Mixed Precision: {amp}
    """
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
    criterion = nn.MSELoss()
    grad_scaler = torch.amp.GradScaler(enabled=amp)
    global_step = 0

    # 5. Train the model
    logger.info("Training started")
    model = model.to(device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            x, y = batch
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            with torch.amp.autocast(device_type=str(device), enabled=amp):
                y_pred = model(x)
                loss = criterion(y_pred, y)
                epoch_loss += loss.item()

            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            logger.info(
                f"Train Step: {global_step:<5d} | Loss: {loss.item():<2.8f} | Epoch: {epoch}"
            )

            # Evaluation round
            if global_step % 100 == 0:
                val_score = evaluate(model, val_loader, device)
                scheduler.step(val_score)
                logger.info(f"Validation Score: {val_score}")

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, f"{dir_checkpoint}/checkpoint_epoch{epoch}.pth")
            logger.info(f"Checkpoint {epoch} saved !")


def evaluate(model, val_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()

    return val_loss / len(val_loader)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=3)
    train(model=model, device=device, epochs=5, batch_size=1, learning_rate=1e-5)

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pathlib import Path
from dvclive import Live
from tqdm import tqdm


from unet import UNet
from utils.dataset import CIIDataset

pre_folder = "./CIII/pre/front/processed"
post_folder = "./CIII/post/front/processed"
dir_checkpoint = "./checkpoints/"
time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def train(
    model,
    device,
    epochs: int,
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_size: float = 448,
    amp: bool = False,
    weight_decay: float = 1e-8,
):
    live = Live(report=None)
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

    live.log_params(
        {
            "device": device.type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "training_size": n_train,
            "validation_size": n_val,
            "images_size": img_size,
            "mixed_precision": amp,
            "save_checkpoint": save_checkpoint,
        }
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
    criterion = nn.MSELoss()
    grad_scaler = torch.amp.GradScaler(enabled=amp)

    # 5. Train the model
    model = model.to(device=device)
    for epoch in range(1, epochs + 1):
        live.log_metric(
            "learning_rate", val=optimizer.param_groups[0]["lr"], timestamp=True
        )

        model.train()
        epoch_loss = 0
        last_val_loss = 0

        for batch in tqdm(
            train_loader, desc=f"Epoch [{epoch:>{len(str(epochs))}}/{epochs}]"
        ):
            x, y = batch
            x = x.to(device=device, dtype=torch.float32)
            # print(x.shape) # [1, 3, 128, 128]
            y = y.to(device=device, dtype=torch.float32)
            # print(y.shape) # [1, 3, 128, 128]

            with torch.amp.autocast(device_type=str(device), enabled=amp):
                y_pred = model(x)
                # print(y_pred.shape) # [1, 3, 128, 128]
                loss = criterion(y_pred, y)
                epoch_loss += loss.item()

            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            live.log_metric("train_loss_step", loss.item())
            live.next_step()

        live.log_metric("train_loss", epoch_loss / len(train_loader))

        # Evaluation
        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        live.log_metric("val_loss", val_loss)

        if save_checkpoint and val_loss < last_val_loss:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            save_path = f"{dir_checkpoint}/checkpoint_{time_str}.pth"
            torch.save(state_dict, save_path)
            live.log_param("checkpoint_epoch", epoch)
            live.log_artifact(save_path, type="model")

        last_val_loss = val_loss

    live.end()


def evaluate(model, val_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
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
    train(model=model, device=device, epochs=100)

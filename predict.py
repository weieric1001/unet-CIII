import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from unet import UNet


def predict(net, device, img, transform=None):
    net.eval()
    net = net.to(device=device)

    with torch.no_grad():
        if transform:
            img = transform(img)
        if len(img.shape) == 3:  # [C, H, W]
            img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)
        output = net(img)  # [N, C, H, W]
        if output.shape[-2:] != img.shape[-2:]:
            output = F.interpolate(
                output, size=img.shape[-2:], mode="bilinear", align_corners=False
            )
    return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=3)
    model.load_state_dict(
        torch.load("./checkpoints/checkpoint_epoch7.pth", weights_only=True)
    )
    model.eval()
    input_image = Image.open("./CIII/pre/front/processed/625505.jpg")
    ground_truth = Image.open("./CIII/post/front/processed/625505.jpg")
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    output = predict(model, device, input_image, transform).squeeze()
    ground_truth = transform(ground_truth)
    criterion = nn.MSELoss()
    loss = criterion(output, ground_truth.to(device))
    print(f"Loss: {loss.item()}")

    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    ground_truth = ground_truth.cpu() * std + mean
    ground_truth = torch.clamp(ground_truth, 0, 1)
    output = output.cpu() * std + mean
    output = torch.clamp(output, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].set_title("Input Image")
    axes[0].imshow(input_image.resize((256, 256), Image.Resampling.BILINEAR))
    axes[1].set_title("Ground Truth")
    axes[1].imshow(ground_truth.detach().permute(1, 2, 0))
    axes[2].set_title("Prediction")
    axes[2].imshow(output.detach().permute(1, 2, 0))
    plt.show()

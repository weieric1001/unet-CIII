import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

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
        output = net(img)
        if output.shape[-2:] != img.shape[-2:]:
            output = F.interpolate(
                output, size=img.shape[-2:], mode="bilinear", align_corners=False
            )
        img_pred = output.argmax(dim=1)
        img_pred = img_pred.cpu().numpy()
    return img_pred


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=3)
    model.load_state_dict(torch.load("./checkpoints/checkpoint_epoch1.pth"))
    model.eval()
    img = Image.open("./CIII/pre/front/processed/625505.jpg")
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    output = predict(model, device, img, transform)

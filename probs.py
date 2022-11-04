import torch
import torchvision
from torchvision import transforms
from model import Net
from PIL import Image


def main():
    # ちゃんと推論できない、画像の読み込み or 前処理で失敗している
    image = Image.open("probs.drawio.png")
    model_path = "./result/latest.pth"
    device = torch.device("mps")
    net = Net()
    net = net.to(device)
    net.load_state_dict(torch.load(model_path))

    trans = transforms.Compose(
        [transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor()]
    )
    inputs = trans(image)
    viewimage = transforms.ToPILImage()
    viewimage(inputs).show()
    inputs = inputs.unsqueeze(0).to(device)

    net.eval()
    outputs = net(inputs)
    _, predicted = outputs.max(1)
    print(inputs)
    print(outputs)
    print(f"{predicted[0]}")


if __name__ == "__main__":
    main()

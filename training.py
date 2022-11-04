import torch
import torch.nn as nn
import torch.optim as optim
from mnist import get_dataset
from model import Net
import matplotlib.pyplot as plt  # グラフ出力用module
import os
from pathlib import Path

BATCH_SIZE = 100
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
EPOCH = 25


def get_device(device_type):
    if device_type is not None:
        return torch.device(device_type)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def main():
    trainloader, testloader = get_dataset("./data")

    # GPUの定義
    device = get_device()

    # モデルの定義
    net = Net()
    net = net.to(device)

    # 誤差関数の定義
    criterion = nn.CrossEntropyLoss()

    # 最適化手法の定義
    optimizer = optim.SGD(
        net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY
    )
    # 学習
    loss_value, acc_value, net = training(
        device, net, trainloader, testloader, criterion, optimizer
    )

    # モデルの保存
    if not os.path.isdir("./result"):
        os.mkdir("./result")
    p = Path("./result")
    torch.save(net.state_dict(), str(p / "latest.pth"))

    # 可視化
    output_graph(loss_value, acc_value)


def training(device, net, trainloader, testloader, criterion, optimizer):
    loss_value = []  # testのlossを保持するlist
    acc_value = []  # testのaccuracyを保持するlist

    for epoch in range(EPOCH):
        # トレーニング
        for (inputs, labels) in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        sum_loss = 0.0
        sum_correct = 0
        sum_total = 0

        # テスト
        with torch.no_grad():
            for (inputs, labels) in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                sum_loss += loss.item()
                _, predicted = outputs.max(1)
                sum_total += labels.size(0)
                sum_correct += (predicted == labels).sum().item()

        loss_rate = sum_loss * BATCH_SIZE / len(testloader.dataset)
        accuracy = float(sum_correct / sum_total)
        print(f"[{epoch + 1:3d}] loss={loss_rate:8.6f}, accuracy={accuracy:8.6f}")

        loss_value.append(sum_loss * BATCH_SIZE / len(testloader.dataset))
        acc_value.append(float(sum_correct / sum_total))

    return loss_value, acc_value, net


def output_graph(loss_value, acc_value, path="./result"):
    # グラフ描画用
    plt.figure(figsize=(6, 6))

    if not os.path.isdir(path):
        os.mkdir(path)
    p = Path(path)

    xlim = len(loss_value)
    # 以下グラフ描画
    plt.plot(range(xlim), loss_value)
    plt.xlim(0, xlim)
    plt.ylim(0, 2.5)
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS")
    plt.legend(["loss"])
    plt.title("loss")
    plt.savefig(str(p / "loss_image.png"))
    plt.clf()

    xlim = len(acc_value)
    plt.plot(range(xlim), acc_value)
    plt.xlim(0, xlim)
    plt.ylim(0, 1)
    plt.xlabel("EPOCH")
    plt.ylabel("ACCURACY")
    plt.legend(["acc"])
    plt.title("accuracy")
    plt.savefig(str(p / "accuracy_image.png"))


if __name__ == "__main__":
    main()

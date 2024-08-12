import time
import copy
import torch
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas
from model import LeNet

"""处理训练集和验证集"""


def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=2)
    val_dataloader = Data.DataLoader(dataset=train_data,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=2)

    return train_dataloader, val_dataloader


"""训练模型"""


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 设定训练的设备 macos为mps
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 定义优化器  Adam可以理解为梯度下降的变种
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    """回归一般使用均方差，分类一般使用交叉熵损失"""
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 将模型放入训练设备当中
    model = model.to(device)

    # 复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Eopch: {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # 初始化参数
        # 训练集损失函数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0

        # 验证集损失函数
        val_loss = 0.0
        # 验证集精确度
        val_corrects = 0

        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        for step, (images, labels) in enumerate(train_dataloader):
            # print("step: {}/{}".format(step, len(train_dataloader)))
            # 将特征值放入训练设备
            images = images.to(device)
            # 将标签放入训练设备
            labels = labels.to(device)
            # 设置模型为训练模式
            model.train()

            # 前向传播过程，输入为一个批次的数据，输出为一个批次数据对应的预测（输出为一个向量）
            output = model(images)

            # 找出每一批次中最大值对应的行标   也就是找到每个图片的最大概率的分类
            pre_lab = torch.argmax(output, dim=1)

            # 通过输出与对应的标签计算损失函数
            loss = criterion(output, labels)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 根据反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加   loss为每个样本的平均loss值
            train_loss += loss.item() * images.size(0)
            # 如果预测正确，则准确度+1
            train_corrects += torch.sum(pre_lab == labels.data)
            # 当前训练的样本数量
            train_num += images.size(0)

        for step, (images, labels) in enumerate(val_dataloader):
            # print("step: {}/{}".format(step, len(val_dataloader)))
            # 将特征值放入验证设备
            images = images.to(device)
            # 将标签放入验证设备
            labels = labels.to(device)

            # 设置模型为评估模式
            """
            train和eval的区别
            
            train 模式：启用训练专用的功能（如 Dropout、BatchNorm 更新），计算梯度以更新模型参数。
            eval 模式：关闭训练专用的功能，使用固定的参数来执行推理，不计算梯度。
            
            Batch Normalization：在 train 模式下，BatchNorm 层会使用当前批次的数据计算均值和方差，并更新其内部的运行均值和方差统计。
            """
            model.eval()
            # 前向传播过程，输入为一个批次的数据，输出为一个批次数据对应的预测（输出为一个向量）
            output = model(images)

            # 找出每一批次中最大值对应的行标   也就是找到每个图片的最大概率的分类， 这里返回的会是每一个样本得分最高的数组下标
            """
            dim=1 的作用是指定在第一个维度上寻找最大值，并返回该维度上最大值的索引。
            
            假设 output 是一个二维张量（通常在分类任务中，output 是网络的输出，形状为 [batch_size, num_classes]），其中：
            batch_size 是批量的大小，即有多少个样本。
            num_classes 是分类的类别数。
            每一行（即第一个维度 dim=0）代表一个样本的输出，每一列（即第二个维度 dim=1）代表该样本在对应类别上的得分。
            
            假设   output = torch.tensor([[2.5, 3.1, 0.2],
                                         [1.2, 0.7, 4.3],
                                         [0.5, 2.2, 1.9]])
                                         
                  pre_lab会返回如下值
                  
                  pre_lab = torch.tensor([1, 2, 1])

            因为     第一个样本 [2.5, 3.1, 0.2] 中，最大值是 3.1，对应的索引是 1。
                    第二个样本 [1.2, 0.7, 4.3] 中，最大值是 4.3，对应的索引是 2。
                    第三个样本 [0.5, 2.2, 1.9] 中，最大值是 2.2，对应的索引是 1。

            """
            pre_lab = torch.argmax(output, dim=1)

            # 通过输出与对应的标签计算损失函数
            loss = criterion(output, labels)

            # 对损失函数进行累加
            val_loss += loss.item() * images.size(0)
            # 如果预测正确，则准确度+1
            val_corrects += torch.sum(pre_lab == labels.data)
            # 当前验证的样本数量
            val_num += images.size(0)

        # 将该轮次的平均loss值添加到list中 （计算并保存每一次迭代的成本函数（也就是损失函数）和准确率）
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        # train_corrects.float() 这里原先是double，但是mps使用pytorch不能使用double，下面的浮点数同理
        train_acc_all.append(train_corrects.float().item() / train_num)

        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.float().item() / train_num)

        # train_loss_all[-1] 用于访问列表 train_loss_all 中的最后一个元素。在 Python 中，列表支持负索引，-1 表示列表中的最后一个元素，-2 表示倒数第二个元素，依此类推。
        print("epoch: {}, train_loss: {:.4f}  train_acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("epoch: {}, val_loss: {:.4f}  val_acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度的权重参数
        if val_acc_all[-1] > best_acc:
            # 保存最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前模型参数
            best_model_wts = copy.deepcopy(model.state_dict())
        # 训练耗时
        time_use = time.time() - since
        print("Training complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优参数
    # 加载最高准确率的模型参数
    torch.save(best_model_wts, "./best_model.pth")

    train_process = pandas.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all
    })

    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, "ro-", label="train_loss_all")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "bs-", label="val_loss_all")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, "ro-", label="train_acc_all")
    plt.plot(train_process["epoch"], train_process.val_acc_all, "bs-", label="val_acc_all")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 实例化模型
    LeNet = LeNet()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(LeNet, train_dataloader, val_dataloader, 20)
    matplot_acc_loss(train_process)
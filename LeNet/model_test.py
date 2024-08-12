import torch
import torch.utils.data as Data
from torchinfo import summary
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from model import LeNet

def test_data_process():
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       num_workers=0)

    return test_dataloader

def test_model_process(model, test_dataloader):
    # 设定训练的设备 macos为mps
    device = "mps" if (torch.backends.mps.is_available()) else "cpu"
    # 将模型放入设备当中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播，不计算梯度，从而节省资源，加快运算速度
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 设置为评估模式
            model.eval()
            # 前向传播过程，输入为测试数据集，输出为每个样本的预测值
            output = model(images)
            # 查找每一行最大值的下标
            pre_label = torch.argmax(output, dim=1)

            # 如果预测正确test_corrects + 1
            test_corrects += torch.sum(pre_label == labels.data)
            # 将测试样本进行累加
            test_num += images.size(0)

    # 计算正确率
    test_acc = test_corrects.float().item() / test_num
    print("test_acc:{}".format(test_acc))

if __name__ == '__main__':
    # 加载模型
    model = LeNet()

    model.load_state_dict(torch.load('./best_model.pth'))

    # 测试正确率
    # test_model_process(model, test_data_process())

    # 设定训练的设备 macos为mps
    device = "mps" if (torch.backends.mps.is_available()) else "cpu"
    model = model.to(device)

    test_data_loader = test_data_process()


    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 设置验证模式
            model.eval()
            output = model(images)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = labels.item()

            print("result:{}, label:{}".format(classes[result], classes[label]))
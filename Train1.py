import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from Model import modelInter

# Train1.py is not 100% correct.

def get_model():
    MDL = modelInter()
    model = MDL.get_cls_model()

    return MDL.get_cls_model()

# 设置随机种子
torch.manual_seed(42)

# 定义超参数
batch_size = 64
learning_rate = 0.0001
num_epochs = 100

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
train_dataset = ImageFolder(r"./dataset2/train", transform=transform) # Path to the train dataset
test_dataset = ImageFolder(r"./dataset2/val", transform=transform) # Path to the val dataset

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 加载预训练的ResNet-50模型
model = resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 替换最后一层全连接层，以适应二分类问题

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

best_accuracy = 0

# 训练模型
total_step = len(train_loader)
valacc=[]
train_losses=[]

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item()}")

    epoch_loss=running_loss/total_step
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss}")
    #测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            #print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print("Total: {}, Correct: {}".format(total, correct))
    valacc.append(accuracy)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Accuracy on test images: {accuracy * 100}%")
    if(accuracy > best_accuracy):
        best_accuracy = accuracy
        torch.save(model, './Model1.pth')


model_save_path = './'  # 替换为你想保存模型的位置
train_losses_save_path = './'  # 替换为你想保存训练损失的位置
valacc_save_path = './'  # 替换为你想保存验证准确率的位置


# 保存模型
torch.save(model.state_dict(), model_save_path)

# 保存 train_losses 和 valacc
with open(train_losses_save_path, 'w') as f:
    for loss in train_losses:
        f.write(f"{loss}\n")

with open(valacc_save_path, 'w') as f:
    for acc in valacc:
        f.write(f"{acc}\n")


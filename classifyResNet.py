# #!/usr/bin/python
# # -*- coding:utf-8 -*-
# #
# # 工具类
# import os
# import random
# import shutil
# from shutil import copy2
# from torch import nn
# from torchvision import transforms

# #
# # def data_set_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.1, test_scale=0.1):
# #     '''
# #     读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
# #     :param src_data_folder: 源文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/src_data
# #     :param target_data_folder: 目标文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/target_data
# #     :param train_scale: 训练集比例
# #     :param val_scale: 验证集比例
# #     :param test_scale: 测试集比例
# #     :return:
# #     '''
# #     print("开始数据集划分")
# #     class_names = os.listdir(src_data_folder)
# #     # 在目标目录下创建文件夹
# #     split_names = ['train', 'val', 'test']
# #     for split_name in split_names:
# #         split_path = os.path.join(target_data_folder, split_name)
# #         if os.path.isdir(split_path):
# #             pass
# #         else:
# #             os.mkdir(split_path)
# #         # 然后在split_path的目录下创建类别文件夹
# #         for class_name in class_names:
# #             class_split_path = os.path.join(split_path, class_name)
# #             if os.path.isdir(class_split_path):
# #                 pass
# #             else:
# #                 os.mkdir(class_split_path)
# #
# #     # 按照比例划分数据集，并进行数据图片的复制
# #     # 首先进行分类遍历
# #     for class_name in class_names:
# #         current_class_data_path = os.path.join(src_data_folder, class_name)
# #         current_all_data = os.listdir(current_class_data_path)
# #         current_data_length = len(current_all_data)
# #         current_data_index_list = list(range(current_data_length))
# #         random.shuffle(current_data_index_list)
# #
# #         train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
# #         val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
# #         test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
# #         train_stop_flag = current_data_length * train_scale
# #         val_stop_flag = current_data_length * (train_scale + val_scale)
# #         current_idx = 0
# #         train_num = 0
# #         val_num = 0
# #         test_num = 0
# #         for i in current_data_index_list:
# #             src_img_path = os.path.join(current_class_data_path, current_all_data[i])
# #             if current_idx <= train_stop_flag:
# #                 copy2(src_img_path, train_folder)
# #                 # print("{}复制到了{}".format(src_img_path, train_folder))
# #                 train_num = train_num + 1
# #             elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
# #                 copy2(src_img_path, val_folder)
# #                 # print("{}复制到了{}".format(src_img_path, val_folder))
# #                 val_num = val_num + 1
# #             else:
# #                 copy2(src_img_path, test_folder)
# #                 # print("{}复制到了{}".format(src_img_path, test_folder))
# #                 test_num = test_num + 1
# #
# #             current_idx = current_idx + 1
# #
# #         print("*********************************{}*************************************".format(class_name))
# #         print(
# #             "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale, current_data_length))
# #         print("训练集{}：{}张".format(train_folder, train_num))
# #         print("验证集{}：{}张".format(val_folder, val_num))
# #         print("测试集{}：{}张".format(test_folder, test_num))
# #
# #
# # if __name__ == '__main__':
# #     src_data_folder = r"C:\Users\Jifeng.LI\Desktop\dataall\datasetIn4"
# #     target_data_folder = r"C:\Users\Jifeng.LI\Desktop\targetdata"
# #     data_set_split(src_data_folder, target_data_folder)
#
#
# #import os
#
#
# # def generate(dir, label):
# #     files = os.listdir(dir)
# #     files.sort()
# #     print('****************')
# #     print('input :', dir)
# #     print('start...')
# #     listText = open('test.txt', 'a')
# #     for file in files:
# #         fileType = os.path.split(file)
# #         if fileType[1] == '.txt':
# #             continue
# #         name = file + ' ' + str(int(label)) + '\n'
# #         listText.write(name)
# #     listText.close()
# #     print('down!')
# #     print('****************')
# #
# #
# # outer_path = r'C:\Users\Jifeng.LI\Desktop\targetdata\test'  # 这里是你的图片的目录
# #
# # if __name__ == '__main__':
# #     i = 0
# #     folderlist = os.listdir(outer_path)  # 列举文件夹
# #     for folder in folderlist:
# #         generate(os.path.join(outer_path, folder), i)
# #         i += 1
#
# from PIL import Image
# from torch.utils.data import Dataset
#
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(256),
#         # 转换成tensor向量
#         transforms.ToTensor(),
#         # 对图像进行归一化操作
#         # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(256),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }
#
# class my_Data_Set(nn.Module):
#     def __init__(self, txt, transform=None, target_transform=None, loader=None):
#         super(my_Data_Set, self).__init__()
#         # 打开存储图像名与标签的txt文件
#         fp = open(txt, 'r')
#         images = []
#         labels = []
#         # 将图像名和图像标签对应存储起来
#         for line in fp:
#             line.strip('\n')
#             line.rstrip()
#             information = line.split()
#             images.append(information[0])
#             labels.append(int(information[1]))
#         self.images = images
#         self.labels = labels
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader
#
#     # 重写这个函数用来进行图像数据的读取
#     def __getitem__(self, item):
#         # 获取图像名和标签
#         imageName = self.images[item]
#         label = self.labels[item]
#         # 读入图像信息
#         image = self.loader(imageName)
#         # 处理图像数据
#         if self.transform is not None:
#             image = self.transform(image)
#         return image, label
#
#     # 重写这个函数，来看数据集中含有多少数据
#     def __len__(self):
#         return len(self.images)
#
#
# # 生成Pytorch所需的DataLoader数据输入格式
# train_dataset = my_Data_Set('train.txt', transform=data_transforms['train'], loader=Load_Image_Information)
# test_dataset = my_Data_Set('val.txt', transform=data_transforms['val'], loader=Load_Image_Information)
# train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import torchvision.transforms as visiontransforms
from time_utils import time_for_file, print_log
from visualiztion import draw_loss_and_accuracy
import random
import numpy as np
import os.path as osp
import time

log_save_root_path = "./"
model_save_root_path = "./"
train_path = r"C:\Users\27654\Desktop\AI WORK\datasetOfJobSurface"
test_path = r"C:\Users\27654\Desktop\AI WORK\datasetOfJobSurface"


def loadTrainData():
    vision_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trainset = torchvision.datasets.ImageFolder(train_path, transform=transforms.Compose(
        [visiontransforms.RandomResizedCrop(224),
         visiontransforms.RandomHorizontalFlip(),
         visiontransforms.ToTensor(),
         vision_normalize]
    ))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    return trainloader


def loadTestData():
    vision_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    testset = torchvision.datasets.ImageFolder(test_path, transform=transforms.Compose(
        [visiontransforms.Resize(256),
         visiontransforms.CenterCrop(224),
         visiontransforms.ToTensor(),
         vision_normalize]
    ))
    testloader = DataLoader(testset, batch_size=32)
    return testloader


def adjust_learning_rate(optimizer, epoch, train_epoch, learning_rate):
    lr = learning_rate * (0.6 ** (epoch / train_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target):
    output = np.array(output.numpy())
    target = np.array(target.numpy())
    prec = 0
    for i in range(output.shape[0]):
        print(output[i])
        pos = np.unravel_index(np.argmax(output[i]), output.shape)
        pre_label = pos[1]
        print(pre_label,target[i])
        if pre_label == target[i]:
            prec += 1
    prec /= target.size
    prec *= 100
    return prec


def resnet_finute(train_epoch, print_freq, learning_rate_start):
    log = open(osp.join(log_save_root_path, 'cluster_seed_{}_{}.txt'.format(random.randint(1, 10000), time_for_file())),
               'w')
    net = models.resnet50(pretrained=True)
    channel_in = net.fc.in_features
    class_num = 4
    net.fc = nn.Sequential(
        nn.Linear(channel_in, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, class_num),
        nn.LogSoftmax(dim=1)
    )
    for param in net.parameters():
        param.requires_grad = False

    for param in net.fc.parameters():
        param.requires_grad = True

    # # 输出网络的结构
    # for child in net.children():
    #     print(child)

    # net.load_state_dict(
    #     torch.load(osp.join(model_save_root_path, 'resnet50_50_2020-04-09_22-15-11.pth')))
    # 后来训练100轮的模型是在我之前训练完50轮的基础上训练的，如果想从头开始训练，可以把加载参数这一句注释掉，但是可能需要重新调整学习率和衰减率

    # 用于可视化Loss和Accuracy的列表
    Loss_list = []
    Accuracy_list = []

    trainloader = loadTrainData()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate_start, momentum=0.9)
    criterion = nn.NLLLoss()
    print_log('Training dir : {:}'.format(train_path), log)
    for epoch in range(train_epoch):
        epoch_accuracy = 0
        epoch_loss = 0
        learning_rate = adjust_learning_rate(optimizer, epoch, train_epoch, learning_rate_start)
        print_log('epoch : [{}/{}] lr={}'.format(epoch, train_epoch, learning_rate), log)
        net.train()
        for i, (inputs, target) in enumerate(trainloader):
            print(target)
            output = net(inputs)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec = accuracy(output.data, target)
            epoch_accuracy += prec
            epoch_loss += loss
            if i % print_freq == 0 or i + 1 == len(trainloader):
                print_log(
                    'after {} epoch, {}th batchsize, prec:{}%,loss:{},input:{},output:{}'.format(epoch + 1, i + 1, prec,
                                                                                                 loss, inputs.size(),
                                                                                                 output.size()), log)
        epoch_loss /= len(trainloader)
        epoch_accuracy /= len(trainloader)
        Loss_list.append(epoch_loss)
        Accuracy_list.append(epoch_accuracy)
        torch.save(net.state_dict(), osp.join(model_save_root_path, 'resnet50_{}_{}.pth'.format(epoch + 1,
                                                                                                time.strftime(
                                                                                                    "%Y-%m-%d_%H-%M-%S"))))
    draw_loss_and_accuracy(Loss_list, Accuracy_list, train_epoch)


def resnet_eval(single_image=False, img_path=None):
    resnet = models.resnet50(pretrained=True)
    channel_in = resnet.fc.in_features
    class_num = 4
    resnet.fc = nn.Sequential(
        nn.Linear(channel_in, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, class_num),
        nn.LogSoftmax(dim=1)
    )
    resnet.load_state_dict(
        torch.load("resnet50_51_2022-10-24_04-44-52.pth",map_location='cpu'))  # 这里填最新的模型的名字
    if single_image == False:
        resnet.eval()
        val_loader = loadTestData()
        criterion = torch.nn.CrossEntropyLoss()
        sum_accuracy = 0
        for i, (inputs, target) in enumerate(val_loader):
            with torch.no_grad():
                output = resnet(inputs)
                loss = criterion(output, target)
                prec = accuracy(output.data, target)

                print(prec)
                sum_accuracy += prec
                print('for {}th batchsize, Eval:Accuracy:{}%,loss:{},input:{},output:{}'.format(i + 1, prec, loss,


                                                                                                inputs.size(),
                                                                                                output.size()))
        sum_accuracy /= len(val_loader)
        print('sum of accuracy = {}'.format(sum_accuracy))
    else:
        resnet.eval()

        vision_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # vision_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # testset = torchvision.datasets.ImageFolder(test_path, transform=transforms.Compose(
        #     [visiontransforms.Resize(256),
        #      visiontransforms.CenterCrop(224),
        #      visiontransforms.ToTensor(),
        #      vision_normalize]
        # ))

        transform = transforms.Compose(
            [visiontransforms.Resize(256),
             visiontransforms.CenterCrop(224),
             visiontransforms.ToTensor(),
             vision_normalize])
        image_PIL = Image.open(img_path).convert('RGB')
        img_tensor = transform(image_PIL).unsqueeze(0)
        # print(img_tensor)

        result = resnet(img_tensor)
        print(result)
        result = result.detach().numpy()
        result = np.array(result)
        print(result)
        print(np.argmax(result))
        pos = np.unravel_index(np.argmax(result), result.shape)
        print(pos)
        pre_label = pos[1]
        # if pre_label == 0:
        #     pre_label = 'c_rebar'
        # elif pre_label == 1:
        #     pre_label = 'cement'
        # elif pre_label == 2:
        #     pre_label = 'h_rebar'
        # elif pre_label == 3:
        #     pre_label = 'template'
        # else:
        #     pre_label = 'VerticalSteelandScaffolding'
        return pre_label
        print('predicted label is {}'.format(pre_label))


if __name__ == '__main__':
    # train_epoch = 100
    # print_freq = 5
    # learning_rate_start = 0.005
    # resnet_finute(train_epoch, print_freq, learning_rate_start)
    # resnet_eval()
    resnet_eval(True, 'img_1.png')
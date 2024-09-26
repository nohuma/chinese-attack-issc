import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import argparse
from torchvision import transforms
from torch.autograd import Variable
import cv2
import numpy as np


class BaseLoader(torch.utils.data.Dataset):
    def __init__(self, triplets, transform=None):
        self.triplets = triplets
        self.transform = transform

    def __getitem__(self, index):
        img1_pth, img2_pth, img3_pth = self.triplets[index]
        img1 = cv2.imread(img1_pth)
        img2 = cv2.imread(img2_pth)
        img3 = cv2.imread(img3_pth)

        try:
            img1 = cv2.resize(img1, (args.picture_resize, args.picture_resize))
        except Exception as e:
            img1 = np.zeros((args.picture_resize, args.picture_resize, 3), dtype=np.uint8)

        try:
            img2 = cv2.resize(img2, (args.picture_resize, args.picture_resize))
        except Exception as e:
            img2 = np.zeros((args.picture_resize, args.picture_resize, 3), dtype=np.uint8)

        try:
            img3 = cv2.resize(img3, (args.picture_resize, args.picture_resize))
        except Exception as e:
            img3 = np.zeros((args.picture_resize, args.picture_resize, 3), dtype=np.uint8)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)


class BaseCnn(nn.Module):
    def __init__(self):
        super(BaseCnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 8),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 8),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 8),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        return x


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.base = BaseCnn()

    def forward(self, x1, x2, x3):
        x1 = self.base(x1)
        x2 = self.base(x2)
        x3 = self.base(x3)

        return x1, x2, x3


class BaseDset(object):
    def __init__(self):
        self.__base_path = ""

        self.__train_set = {}
        self.__test_set = {}
        self.__train_keys = []
        self.__test_keys = []

    def load(self, base_path):
        """加载数据集，将类别和路径存储"""
        self.__base_path = base_path
        train_dir = os.path.join(self.__base_path, 'train')
        test_dir = os.path.join(self.__base_path, 'test')

        self.__train_set = {}
        self.__test_set = {}
        self.__train_keys = []
        self.__test_keys = []

        for class_id in os.listdir(train_dir):
            # 对于train_dir里的每个文件夹名字 classi
            class_dir = os.path.join(train_dir, class_id)
            # 为其在训练集合中创建一个文件夹
            # 在类别集合中，即train_keys中添加类别classi
            self.__train_set[class_id] = []
            self.__train_keys.append(class_id)
            # 对于每个类别内的数据，将其路径添加到集合中
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.__train_set[class_id].append(img_path)
        # 同理对于测试集合也一样
        for class_id in os.listdir(test_dir):
            class_dir = os.path.join(test_dir, class_id)
            self.__test_set[class_id] = []
            self.__test_keys.append(class_id)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.__test_set[class_id].append(img_path)

        return len(self.__train_keys), len(self.__test_keys)

    # 获取三元组 ！！！
    def getTriplet(self, split='train'):
        # 默认选取训练集
        if split == 'train':
            dataset = self.__train_set
            keys = self.__train_keys
        else:
            dataset = self.__test_set
            keys = self.__test_keys

        # 随机指定两个正负类别，确保二者不一致
        pos_idx = random.randint(0, len(keys) - 1)
        while True:
            neg_idx = random.randint(0, len(keys) - 1)
            if pos_idx != neg_idx:
                break
        # 选定一个原始样本
        pos_anchor_img_idx = random.randint(0, len(dataset[keys[pos_idx]]) - 1)
        # 随机选择一个正样本，保证二者不一致
        while True:
            pos_img_idx = random.randint(0, len(dataset[keys[pos_idx]]) - 1)
            if pos_anchor_img_idx != pos_img_idx:
                break
        # 随机选择一个负样本
        neg_img_idx = random.randint(0, len(dataset[keys[neg_idx]]) - 1)

        # 生成三元组
        pos_anchor_img = dataset[keys[pos_idx]][pos_anchor_img_idx]
        pos_img = dataset[keys[pos_idx]][pos_img_idx]
        neg_img = dataset[keys[neg_idx]][neg_img_idx]

        return pos_anchor_img, pos_img, neg_img


def train(data, model, criterion, optimizer, epoch):
    print("******** Training ********")
    total_loss = 0
    model.train()
    for batch_idx, img_triplet in enumerate(data):
        # 提取数据
        anchor_img, pos_img, neg_img = img_triplet
        anchor_img, pos_img, neg_img = anchor_img.to(device), pos_img.to(device), neg_img.to(device)
        anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
        # 分别获得三个编码
        E1, E2, E3 = model(anchor_img, pos_img, neg_img)
        # 计算二者之间的欧式距离
        dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
        dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

        target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
        target = target.to(device)
        target = Variable(target)
        # 大小如何？
        loss = criterion(dist_E1_E2, dist_E1_E3, target)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 打印一波损失
        log_step = args.train_log_step
        if (batch_idx % log_step == 0) and (batch_idx != 0):
            print('Train Epoch: {} [{}/{}] \t Loss: {:.4f}'.format(epoch, batch_idx, len(data), total_loss / log_step))
            total_loss = 0
    print("****************")


def test(data, model, criterion):
    print("******** Testing ********")
    with torch.no_grad():
        model.eval()
        accuracies = [0, 0, 0]
        acc_threshes = [0, 0.2, 0.5]
        total_loss = 0
        for batch_idx, img_triplet in enumerate(data):
            anchor_img, pos_img, neg_img = img_triplet
            anchor_img, pos_img, neg_img = anchor_img.to(device), pos_img.to(device), neg_img.to(device)
            anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
            E1, E2, E3 = model(anchor_img, pos_img, neg_img)
            dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
            dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

            target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
            target = target.to(device)
            target = Variable(target)

            loss = criterion(dist_E1_E2, dist_E1_E3, target)
            total_loss += loss

            for i in range(len(accuracies)):
                prediction = (dist_E1_E3 - dist_E1_E2 - args.margin * acc_threshes[i]).cpu().data
                prediction = prediction.view(prediction.numel())
                prediction = (prediction > 0).float()
                batch_acc = prediction.sum() * 1.0 / prediction.numel()
                accuracies[i] += batch_acc
        print('Test Loss: {}'.format(total_loss / len(data)))
        for i in range(len(accuracies)):
            # 0%等价于准确率其余是更严格的指标
            print(
                'Test Accuracy with diff = {}% of margin: {:.4f}'.format(acc_threshes[i] * 100,
                                                                         accuracies[i] / len(data)))
    print("****************")

    return accuracies[-1]


def main():
    # random_seed
    torch.manual_seed(718)
    torch.cuda.manual_seed(718)

    data_path = r'./characters'
    # data_path = r'./characters'
    dset_obj = BaseDset()
    dset_obj.load(data_path)

    train_triplets = []
    test_triplets = []

    for i in range(args.num_train_samples):
        pos_anchor_img, pos_img, neg_img = dset_obj.getTriplet()
        train_triplets.append([pos_anchor_img, pos_img, neg_img])
    for i in range(args.num_test_samples):
        pos_anchor_img, pos_img, neg_img = dset_obj.getTriplet(split='test')
        test_triplets.append([pos_anchor_img, pos_img, neg_img])
    loader = BaseLoader
    model = SiameseNet()
    model.to(device)

    criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc_of_50_margin = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # 初始化数据加载器
        # 加载三元组
        train_data_loader = torch.utils.data.DataLoader(
            loader(train_triplets,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(0, 1)
                   ])),
            batch_size=args.batch_size, shuffle=True)
        test_data_loader = torch.utils.data.DataLoader(
            loader(test_triplets,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(0, 1)
                   ])),
            batch_size=args.batch_size, shuffle=True)
        train(train_data_loader, model, criterion, optimizer, epoch)
        acc_of_50_margin = test(test_data_loader, model, criterion)

        model_to_save = {
            "epoch": epoch + 1,
            'state_dict': model.state_dict(),
        }

        if acc_of_50_margin > best_acc_of_50_margin:
            best_acc_of_50_margin = acc_of_50_margin
            best_epoch = epoch

            if not args.disable_save_best_ckp:
                result_path = os.path.join(args.result_dir)
                file_name = os.path.join(args.result_dir, "best_checkpoint" + ".pt")
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                save_checkpoint(model_to_save, file_name)

        if (epoch % args.ckp_freq == 0) and not args.disable_save_ckp:
            result_path = os.path.join(args.result_dir)
            file_name = os.path.join(args.result_dir, "checkpoint_" + str(epoch) + ".pt")
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            save_checkpoint(model_to_save, file_name)
    print("Training is done.")
    print(f"The best epoch of acc50, which is {best_acc_of_50_margin * 100}%, is {best_epoch}.")


def save_checkpoint(state, file_name):
    torch.save(state, file_name)


if __name__ == '__main__':
    # 超参数
    parser = argparse.ArgumentParser(description='PyTorch Siamese Example')
    parser.add_argument('--result_dir', default='output', type=str,
                        help='Directory to store results')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument("--disable_save_ckp", default=False, action='store_true',
                        help="disable to save checkpoint frequently")
    parser.add_argument('--ckp_freq', type=int, default=5, metavar='N',
                        help='Checkpoint Frequency (default: 1)')
    parser.add_argument("--disable_save_best_ckp", default=False, action='store_true',
                        help="disable to save best checkpoint")

    parser.add_argument('--train_log_step', type=int, default=500, metavar='M',
                        help='Number of iterations after which to log the loss')
    parser.add_argument('--margin', type=float, default=1.0, metavar='M',
                        help='margin for triplet loss (default: 1.0)')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='M',
                        help='Dataset (default: mnist)')
    parser.add_argument('--picture_resize', type=int, default=200, metavar='M',
                        help='size of the picture to reset (default: 200)')
    parser.add_argument('--num_train_samples', type=int, default=50000, metavar='M',
                        help='number of training samples (default: 50000)')
    parser.add_argument('--num_test_samples', type=int, default=10000, metavar='M',
                        help='number of test samples (default: 10000)')

    global args, device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()

    main()

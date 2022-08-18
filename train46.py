import os
from re import T
import time
import datetime
import easydict
import random
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader


args = easydict.EasyDict({
    # device setting
    'device': 0,
    'seed' : 123,
    
    # training setting
    'batch_size' : 64,
    'num_workers' : 0,
    'epoch' : 30,
    
    # optimizer & criterion
    'lr' : 0.01,
    'momentum' : 0.9,
    'weight_decay' : 1e-4,
    'nesterov' : True,
    
    # directory
    'data_path' : './dataset',
    'save_path' : './save',
    # etc
    'print_freq' : 30,
    'threshold' : 0.5,
})

def setup(args):
    device = torch.device("cuda")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)
    return device

def load_data(args, data_type:str='train'):
    """
    data_type(str): train or test
    """
    start = time.time()
    data_path = Path(args.data_path) / data_type
    features = np.load(data_path/'features.npy')
    if data_type == 'test':
        labels = np.zeros_like(features)  # dummy test label
    else:
        labels = np.load(data_path/'labels.npy')
    end = time.time()
    sec = end - start
    print(f"Completed Loading {data_type} data at {str(datetime.timedelta(seconds=sec)).split('.')[0]}")
    return features, labels



train_data, train_label = load_data(args, 'train')
test_data, test_label = load_data(args, 'test')



args.num_features = train_data.shape[1]
args.num_classes = len(np.unique(train_label))
print(f"Train_shape: {train_data.shape}")
print(f"Test_shape: {test_data.shape}")
print(f"Number of classes: {len(np.unique(train_label))}")


#check label_data shape
print(train_label.shape)

class Dataset:
    def __init__(self, features, labels, transform=None):
        """Basic Dataset Class
        
        :arg
            features: numpy array(features)
            labels: numpy asrray(labels)
        """
        self.features = features
        self.labels = labels
        self.classes = np.unique(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        
        if self.transform:
            feature = self.transform(feature)
        
        label = self.labels[idx]
        return feature, label

class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(4096)
        self.relu = nn.ReLU(inplace=True)
        #self.pool = nn.MaxPool2d(2)
        #self.drop = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(in_channels=4096, out_channels=1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.conv3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=4096, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(4096)
        #self.drop = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    
        self.fc1 = nn.Linear(4096, 2048, bias = True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 2048, bias = True)
        self.fc3 = nn.Linear(2048, 80, bias = True)
       # self.fc4 = nn.Linear(80, 80, bias = True)


        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.pool(x)
        # x = self.drop(x)
        
        x = self.conv2(x)
        x = self.bn2(x)   
        x = self.relu(x)
        
        
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.relu(x)
#        x = self.pool(x)
 #       x = self.drop(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1) 

        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


class AlexNet(nn.Module):
  def __init__(self, num_classes=1000):
    super().__init__()
    ##### CNN layers 
    self.net = nn.Sequential(
        # conv1
        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
        nn.ReLU(inplace=True),  # non-saturating function
        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # 논문의 LRN 파라미터 그대로 지정
        nn.MaxPool2d(kernel_size=3, stride=2),
        # conv2
        nn.Conv2d(96, 256, kernel_size=5, padding=2), 
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # conv3
        nn.Conv2d(256, 384, 3, padding=1),
        nn.ReLU(inplace=True),
        # conv4
        nn.Conv2d(384, 384, 3, padding=1),
        nn.ReLU(inplace=True),
        # conv5
        nn.Conv2d(384, 256, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),

    )

    ##### FC layers
    self.classifier = nn.Sequential(
        # fc1
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
        nn.ReLU(inplace=True).
        # fc2
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )
    # bias, weight 초기화 
    def init_bias_weights(self):
      for layer in self.net:
        if isinstance(layer, nn.Conv2d):
          nn.init.normal_(layer.weight, mean=0, std=0.01)   # weight 초기화
          nn.init.constant_(layer.bias, 0)   # bias 초기화
      # conv 2, 4, 5는 bias 1로 초기화 
      nn.init.constant_(self.net[4].bias, 1)
      nn.init.constant_(self.net[10].bias, 1)
      nn.init.constant_(self.net[12].bias, 1)
    # modeling 
    def forward(self, x):
      x = self.net(x)   # conv
      x = x.view(-1, 256*6*6)   # keras의 reshape (텐서 크기 2d 변경)
      return self.classifier(x)   # fc   


class Metric:
    def __init__(self, header='', fmt='{val:.4f} ({avg:.4f})'):
        """Base Metric Class 
        :arg
            fmt(str): format representing metric in string
        """
        self.val = 0
        self.sum = 0
        self.n = 0
        self.avg = 0
        self.header = header
        self.fmt = fmt

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach().clone()

        self.val = val
        self.sum += val * n
        self.n += n
        self.avg = self.sum / self.n

    def compute(self):
        return self.avg

    def __str__(self):
        return self.header + ' ' + self.fmt.format(**self.__dict__)
    
def train_one_epoch(model, train_dataloader, optimizer, criterion, epoch, args):
    # 1. create metric
    data_m = Metric(header='Data:')
    batch_m = Metric(header='Batch:')
    loss_m = Metric(header='Loss:')

    # 2. start validate
    model.train()

    total_iter = len(train_dataloader)
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(train_dataloader):
        batch_size = x.size(0)

        x = x.to(args.device)
        y = y.to(args.device)

        data_m.update(time.time() - start_time)

        y_hat = model(x)
        loss = criterion(y_hat, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_m.update(loss, batch_size)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            print(f"TRAIN({epoch:03}): [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m} {loss_m}")

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    # 3. calculate metric
    duration = str(datetime.timedelta(seconds=batch_m.sum)).split('.')[0]
    data = str(datetime.timedelta(seconds=data_m.sum)).split('.')[0]
    f_b_o = str(datetime.timedelta(seconds=batch_m.sum - data_m.sum)).split('.')[0]
    loss = loss_m.compute()

    # 4. print metric
    space = 16
    num_metric = 5
    print('-'*space*num_metric)
    print(("{:>16}"*num_metric).format('Stage', 'Batch', 'Data', 'F+B+O', 'Loss'))
    print('-'*space*num_metric)
    print(f"{'TRAIN('+str(epoch)+')':>{space}}{duration:>{space}}{data:>{space}}{f_b_o:>{space}}{loss:{space}.4f}")
    print('-'*space*num_metric)

    return loss

def prediction_mask(output, args):
    """
    학습된 모델이 예측한 값이 Threshold 값을 넘지 못하면 Train에 없던 데이터로 판단하여 -1 라벨로 예측하도록 지정
    """
    prediction = torch.argsort(output, dim=-1, descending=True)
    mask = F.softmax(output).max(dim=-1)[0] < args.threshold
    prediction[mask,0] = -1
    prediction = prediction[:, :min(1, output.size(1))].squeeze(-1).tolist()
    return prediction

def prediction_submission(predict, args):
    """
    테스트 데이터의 idx와 예측한 prediction label을 submission.csv로 저장
    """
    submission = [[idx, label] for idx, label in enumerate(predict)]
    df = pd.DataFrame(data=submission, columns=['id_idx', 'label'], index=None)
    args.save_path = Path(args.save_path)
    args.save_path.mkdir(exist_ok=True)
    df.to_csv(args.save_path / 'submission.csv', index=False)

def test_submission(model, test_dataloader, args):
    """
    학습한 모델로 테스트 데이터의 라벨을 예측하고 그 결과를 submission.csv로 저장
    """
    model_predict = [] # for submission.csv
    for x,y in test_dataloader:
        x = x.to(args.device)
        y = y.to(args.device)
        output = model(x)
        prediction = prediction_mask(output, args)
        model_predict.extend(prediction)
    prediction_submission(model_predict, args)

def run(args):
    start = time.time()
    
    # 1. load train, test dataset
    train_dataset = Dataset(train_data, train_label)
    test_dataset = Dataset(test_data, test_label)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 2. create model
    model = SampleModel().to(args.device)

    # 3. optimizer, criterion
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    #optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 4. train & validate
    for epoch in range(args.epoch):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, epoch, args)
    
    test_submission(model, test_dataloader, args)
    end = time.time()
    sec = end - start
    print(f"Finished Training & Test at {str(datetime.timedelta(seconds=sec)).split('.')[0]} ....")


setup(args)
run(args)
pd.read_csv(args.save_path / 'submission.csv', index_col=False)
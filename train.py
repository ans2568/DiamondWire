# train.py
#!/usr/bin/env	python3

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.CNN import CNN
from model.ensemble import EnsembleNetwork
from model.CNN_residual import CNN_Residual

from conf import settings
from utils.util import make_data_list
from WireDataset import WireDataset, input_transform

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(train_dataloader):
        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(train_dataloader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()
    val_loss = 0.0 # cost function error
    val_correct = 0.0
    for (images, labels) in val_dataloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        val_loss += loss.item()
        _, preds = outputs.max(1)
        val_correct += preds.eq(labels).sum()

    finish = time.time()
    print('GPU INFO.....')
    print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        val_loss / len(val_dataloader.dataset),
        val_correct.float() / len(val_dataloader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', val_loss / len(val_dataloader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', val_correct.float() / len(val_dataloader.dataset), epoch)

    return val_correct.float() / len(val_dataloader.dataset)

if __name__ == '__main__':
    main_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=32, help='batch size')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('-num_workers', type=int, default=8, help='torch DataLoader num_workers')
    parser.add_argument('-net', type=int, default=0, help='select network and dataset(0 ~ 3)')
    parser.add_argument('-model', type=str, help='the first model\'s weights file to train Ensemble')
    parser.add_argument('-model2', type=str, help='the second model\'s weights file to train Ensemble')
    parser.add_argument('-model3', type=str, help='the third model\'s weights file to train Ensemble')
    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')

    networks = [CNN(), CNN_Residual('canny'), CNN_Residual('sobel')]
    networks_name = ['origin_CNN', 'canny_CNN_residual', 'sobel_CNN_residual', 'Ensemble']
    if args.net >= 0 and args.net < 3:
        net = networks[args.net]
    elif args.net == 3:
        net = EnsembleNetwork(args.model, args.model2, args.model3)
    else:
        print('Could not find Model. Please select model between 0 and 3')
        print('-net 0 : CNN')
        print('-net 1 : canny edge preprocessing + CNN + Residual concept')
        print('-net 2 : sobel edge preprocessing + CNN + Residual concept')
        print('-net 3 : Ensemble')
        sys.exit(1)
    net = net.cuda()
    net_name = networks_name[args.net]
    # 경로 선택 dataset/train
    train_path = os.path.join('dataset', 'original_train_test_val', 'train')
    # 경로 선택 dataset/val
    val_path = os.path.join('dataset', 'original_train_test_val', 'val')

    train_data_list = make_data_list(train_path)
    train_dataset = WireDataset(train_data_list, input_transform=input_transform())
    train_dataloader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=args.num_workers)

    val_data_list = make_data_list(val_path)
    val_dataset = WireDataset(val_data_list, input_transform=input_transform())
    val_dataloader = DataLoader(val_dataset, batch_size=args.b, shuffle=False, num_workers=args.num_workers)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, net_name, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, net_name, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 1, 426, 200)
    input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{epoch}-{type}.pth')
    best_acc = 0
    for epoch in range(1, settings.EPOCH + 1):
        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
    writer.close()

    main_finish = time.time()
    print('testing time consumed: {:.2f}s'.format(main_finish - main_start))
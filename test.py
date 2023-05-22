#test.py
#!/usr/bin/env python3

import os
import sys
import time
import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from model.CNN import CNN
from model.ensemble import EnsembleNetwork
from model.CNN_residual import CNN_Residual

from WireDataset import WireDataset, input_transform
from utils.util import make_data_list, save_confusion_matrix, save_csv

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=32, help='batch size')
    parser.add_argument('-num_workers', type=int, default=8, help='torch DataLoader num_workers')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-net', type=int, default=0, help='select network and dataset(0 ~ 3)')
    parser.add_argument('-model', type=str, help='the first model\'s weights file to train Ensemble')
    parser.add_argument('-model2', type=str, help='the second model\'s weights file to train Ensemble')
    parser.add_argument('-model3', type=str, help='the third model\'s weights file to train Ensemble')
    args = parser.parse_args()

    networks = [CNN(), CNN_Residual('canny'), CNN_Residual('sobel')]
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

    path = os.path.join('dataset', 'original_train_test_val', 'test')

    data_list = make_data_list(path)
    dataset = WireDataset(data_list, input_transform=input_transform())
    dataloader = DataLoader(dataset, batch_size=args.b, shuffle=True, num_workers=args.num_workers)

    net.load_state_dict(torch.load(args.weights))
    net.eval()

    correct_1 = 0.0
    total = 0

    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(dataloader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(dataloader)))
            image = image.cuda()
            label = label.cuda()
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')
            output = net(image)
            _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            #compute top1
            correct_1 += correct[:, :1].sum()
            label = label.to('cpu')
            pred = pred.to('cpu')
            true_labels.append(label)
            pred_labels.append(pred)
    finish = time.time()

    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    conf_mat = confusion_matrix(true_labels, pred_labels)
    save_confusion_matrix(conf_mat, ['high', 'medium', 'low'], normalize=True)
    print('GPU INFO.....')
    print(torch.cuda.memory_summary(), end='')

    print()
    accuracy = correct_1 / len(dataloader.dataset)
    accuracy = round(accuracy.item()*100, 2)
    print("Top 1 accuracy: {}%".format(accuracy))
    parameter_num = sum(p.numel() for p in net.parameters())
    print("Parameter numbers: {}".format(parameter_num))
    elapsed_time = round(finish - start, 2)
    print('testing time consumed: {}s'.format(elapsed_time))
    if args.net == 0:
        save_csv(args.b, origin=args.weights, accuracy=accuracy, parameters=parameter_num, elapsed_time=elapsed_time)
    elif args.net == 1:
        save_csv(args.b, canny=args.weights, accuracy=accuracy, parameters=parameter_num, elapsed_time=elapsed_time)
    elif args.net == 2:
        save_csv(args.b, sobel=args.weights, accuracy=accuracy, parameters=parameter_num, elapsed_time=elapsed_time)
    elif args.net == 3:
        save_csv(args.b, origin=args.model, canny=args.model2, sobel=args.model3, ensemble=args.weights, accuracy=accuracy, parameters=parameter_num, elapsed_time=elapsed_time)

import os
import cv2
import csv
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

def canny_transform(images):
    cannys = []
    with torch.no_grad():
        for image in images:
            image_u8 = (image * 255).byte().cpu().numpy().transpose(1, 2, 0). astype(np.uint8)
            canny = cv2.Canny(image=image_u8, threshold1=50, threshold2=115)
            cannys.append(canny)
        imgs = torch.stack([transforms.ToTensor()(img) for img in cannys])
    return imgs

def sobel_transform(images):
    sobels = []
    with torch.no_grad():
        for image in images:
            image_np = (image * 255).byte().cpu().numpy().transpose(1, 2, 0)
            sobelx = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
            sobelxy = cv2.addWeighted(src1=sobelx, alpha=0.5, src2=sobely, beta=0.5, gamma=0).astype(np.float32)
            sobels.append(sobelxy)
        imgs = torch.stack([transforms.ToTensor()(img) for img in sobels])
    return imgs

def scharr_transform(images):
    scharrs = []
    with torch.no_grad():
        for image in images:
            image_np = (image * 255).byte().cpu().numpy().transpose(1, 2, 0)
            scharrx = cv2.Scharr(image_np, cv2.CV_64F, 0, 1)
            scharry = cv2.Scharr(image_np, cv2.CV_64F, 0, 1)
            scharrxy = cv2.addWeighted(src1=scharrx, alpha=0.5, src2=scharry, beta=0.5, gamma=0).astype(np.float32)
            scharrs.append(scharrxy)
        imgs = torch.stack([transforms.ToTensor()(img) for img in scharrs])
    return imgs

def laplacian_transform(images):
    laplacians = []
    with torch.no_grad():
        for image in images:
            image_np = (image * 255).byte().cpu().numpy().transpose(1, 2, 0)
            image_u8 = (image_np * 255).astype(np.uint8)
            laplacian = cv2.Laplacian(image_u8, cv2.CV_64F).astype(np.float32)
            laplacians.append(laplacian)
        imgs = torch.stack([transforms.ToTensor()(img) for img in laplacians])
    return imgs

def make_data_list(root):
    files = os.listdir(root) # 데이터들 dataset/[type]
    data_list = []
    for name in files:
        file = os.path.join(root, name) # 이미지 파일 경로
        data_list.append(file)
    return data_list

def plot_confusion_matrix(mat, labels, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=False):
    plt.figure(figsize=(10, 10))
    plt.imshow(mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks = np.arange(len(labels))
    nlabels = []
    for k in range(len(mat)):
        n = sum(mat[k])
        nlabel = '{0}(n={1})'.format(labels[k], n)
        nlabels.append(nlabel)
    plt.xticks(marks, labels)
    plt.yticks(marks, nlabels)

    thresh = mat.max() / 2
    
    if normalize:
        for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
            plt.text(j, i, '{0}%'.format(mat[i, j] * 100 / sum(mat[i])), horizontalalignment="center", color="white" if mat[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
            plt.text(j, i, mat[i, j], horizontalalignment="center", color="white" if mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def save_confusion_matrix(mat, labels, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=False):
    plt.figure(figsize=(10, 10))
    plt.imshow(mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks = np.arange(len(labels))
    nlabels = []
    for k in range(len(mat)):
        n = sum(mat[k])
        nlabel = '{0}(n={1})'.format(labels[k], n)
        nlabels.append(nlabel)
    plt.xticks(marks, labels)
    plt.yticks(marks, nlabels)

    thresh = mat.max() / 2
    
    if normalize:
        for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
            plt.text(j, i, '{0}%'.format(mat[i, j] * 100 / sum(mat[i])), horizontalalignment="center", color="white" if mat[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
            plt.text(j, i, mat[i, j], horizontalalignment="center", color="white" if mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title + '.png')

def save_csv(batch_size, origin=None, canny=None, sobel=None, ensemble=None, accuracy=None, parameters=None, elapsed_time=None):
    csv_file = 'output_log.csv'
    column = ['batch_size', 'origin', 'canny', 'sobel', 'ensemble', 'accuracy', 'parameters', 'elapsed_time']
    row_data = [batch_size, origin, canny, sobel, ensemble, accuracy, parameters, elapsed_time]
    if os.path.isfile(csv_file):
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row_data)
    else:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column)
            writer.writerow(row_data)
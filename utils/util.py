import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

def make_data_list(root):
    files = os.listdir(root) # 데이터들 dataset/[type]
    data_list = []
    for name in files:
        file = os.path.join(root, name) # 이미지 파일 경로
        data_list.append(file)
    return data_list

def plot_confusion_matrix(mat, labels, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=False):
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

import cv2
import os
from os.path import join, exists
from sklearn.model_selection import train_test_split

"""
    기존 데이터 셋을 train, val, test로 7:2:1의 비율로 분류하는 파일
"""

# 이미지 디렉토리 선택
img_dir = 'dataset'
whole = 'whole/edge_detection/edge_detection_SobelXY'
canny = 'edge_canny_train_test_val'
sobel = 'edge_SobelXY_train_test_val'
scharr = 'edge_ScharrXY_train_test_val'
laplacian = 'edge_Laplacian_train_test_val'
train = 'train'
test = 'test'
val = 'val'

images = [] # 이미지 파일 이름
labels = [] # 이미지 파일 이름에 해당하는 라벨

# 디렉토리가 존재하지 않을 시 디렉토리 생성 함수
def make_dirs(dir, sub_dir):
    dir_path = join(dir, sub_dir)
    if not exists(dir_path):
        os.makedirs(dir_path)

def my_train_test_split(dataset, save_path):
    # files = os.listdir(dataset) 
    files = [f for f in os.listdir(dataset) if os.path.isfile(join(dataset, f))] # 데이터들 dataset/whole/original only file
    for name in files:
        file = join(dataset, name) # 이미지 파일
        label = name.split('_')[0] # high_0.bmp 파일이면 label = high
        # img = cv2.imread(file)
        images.append(name) # 이미지 리스트에 추가
        labels.append(label) # 라벨 리스트에 추가
    
    # 전부 추가한 이미지와 라벨들을 가지고 train_test_split 진행
    # train 7 test 3 로 나눔
    train_img, test_img, train_label, test_label = train_test_split(images, labels, test_size=0.3, random_state=42)
    for file_name in train_img:
        file = join(dataset, file_name) # 이미지 파일
        img = cv2.imread(file)
        cv2.imwrite(join(save_path, train, file_name) , img)
    # test 3 을 val 2 test 1 로 나눔
    val_img, test_img, val_label, test_label = train_test_split(test_img, test_label, test_size=0.33, random_state=42)
    for file_name in test_img:
        file = join(dataset, file_name) # 이미지 파일
        img = cv2.imread(file)
        cv2.imwrite(join(save_path, test, file_name) , img)
    for file_name in val_img:
        file = join(dataset, file_name) # 이미지 파일
        img = cv2.imread(file)
        cv2.imwrite(join(save_path, val, file_name) , img)

if __name__ == '__main__':

    # 기본 Dataset이 존재하는 디렉토리
    original_dataset_path = join(img_dir, whole)

    mkdir_path = join(img_dir, sobel)
    make_dirs(mkdir_path, train)
    make_dirs(mkdir_path, test)
    make_dirs(mkdir_path, val)

    my_train_test_split(original_dataset_path, mkdir_path)
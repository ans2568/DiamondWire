import cv2
from torchvision import transforms
from torch.utils.data import Dataset

def input_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])

class WireDataset(Dataset):
    def __init__(self, data_list, input_transform=None):
        super().__init__()
        self.data_list = data_list
        self.input_transform = input_transform

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # 데이터셋에서 idx 번째 데이터 로드
        image_path = self.data_list[idx]

        # 데이터 Grayscale로 획득
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 이미지 텐서로 변환
        if self.input_transform:
            img_transform = self.input_transform(img)

        label = image_path.split('/')[-1]
        label = label.split('_')[0]
        if label == 'high':
            label = 0
        elif label == 'medium':
            label = 1
        elif label == 'low':
            label = 2

        return img_transform, label
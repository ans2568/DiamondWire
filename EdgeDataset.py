import cv2
from torchvision import transforms
from torch.utils.data import Dataset

def input_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])

class EdgeDataset(Dataset):
    def __init__(self, data_list, edge_filter = 'canny', input_transform=None):
        super().__init__()
        self.data_list = data_list
        self.input_transform = input_transform
        self.type = edge_filter

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # 데이터셋에서 idx 번째 데이터 로드
        image_path = self.data_list[idx]

        # 데이터 Grayscale로 획득
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if self.type == 'sobel':
            # Sobel 필터 적용
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            img = sobelx + sobely
        elif self.type == 'scharr':
            # Scharr 필터 적용
            scharrx = cv2.Scharr(img, cv2.CV_64F, 0, 1)
            scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
            img = scharrx + scharry
        elif self.type == 'laplacian':
            # Laplacian 필터 적용
            img = cv2.Laplacian(img, cv2.CV_64F)
        else:
            # Canny 필터 적용
            # 두 번째 인자가 일정 임계값 보다 낮으면 엣지로 추출 x, 세 번째 인자는 일정 임계값보다 높으면 무조건 엣지로 추출
            img = cv2.Canny(img, 50, 115)

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
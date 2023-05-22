import os
import cv2
from os.path import join
from train_validation_test_split import make_dirs

def edgeDetection(directory):
    # directory 에 해당하는 파일들을 files로 전부 불러옴
    files = [f for f in os.listdir(directory) if os.path.isfile(join(directory, f))]
    for name in files:
        path = join(directory, name)    # dataset/whole/[name]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # Sobel 필터 적용
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobelxy = sobelx + sobely
        cv2.imwrite(join(directory, 'edge_detection_SobelX', name) , sobelx)
        cv2.imwrite(join(directory, 'edge_detection_SobelY', name) , sobely)
        cv2.imwrite(join(directory, 'edge_detection_SobelXY', name) , sobelxy)

        # Scharr 필터 적용
        scharrx = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        scharrxy = scharrx + scharry
        cv2.imwrite(join(directory, 'edge_detection_ScharrX', name) , scharrx)
        cv2.imwrite(join(directory, 'edge_detection_ScharrY', name) , scharry)
        cv2.imwrite(join(directory, 'edge_detection_ScharrXY', name) , scharrxy)

        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        cv2.imwrite(join(directory, 'edge_detection_Laplacian', name) , laplacian)
        # Canny 필터 적용
        # 두 번째 인자가 일정 임계값 보다 낮으면 엣지로 추출 x, 세 번째 인자는 일정 임계값보다 높으면 무조건 엣지로 추출
        cannyxy = cv2.Canny(img, 50, 115)
        cv2.imwrite(join(directory, 'edge_detection_Canny', name) , cannyxy)

path = 'dataset/whole'
make_dirs(path, 'edge_detection_SobelX')
make_dirs(path, 'edge_detection_SobelY')
make_dirs(path, 'edge_detection_SobelXY')

make_dirs(path, 'edge_detection_ScharrX')
make_dirs(path, 'edge_detection_ScharrY')
make_dirs(path, 'edge_detection_ScharrXY')

make_dirs(path, 'edge_detection_Laplacian')
make_dirs(path, 'edge_detection_Canny')

edgeDetection(path)
# Diamond Wire Wear Prediction Model

# Model Architecture

![Ensemble](https://github.com/ans2568/DiamondWire/assets/80823431/b9072e02-481b-498e-9988-9ac57507de8c)

# How to Usage

### Train

1. CNN Network

```bash
python train.py -net 0
```

2. Canny Edge Preprocessing + CNN + Residual Concept

```bash
python train.py -net 1
```

3. Sobel Edge Preprocessing + CNN + Residual Concept

```bash
python train.py -net 2
```

4. Ensemble Network (1+2+3)

```bash
python train.py -net 3 -model path/to/checkpoint/100-regular.pth -model2 path/to/checkpoint/100-regular.pth -model3 path/to/checkpoint/100-regular.pth
```

### Test

1. CNN Network

```bash
python test.py -net 0 -weights path/to/checkpoint/100-regular.pth
```

2. Canny Edge Preprocessing + CNN + Residual Concept

```bash
python test.py -net 1 -weights path/to/checkpoint/100-regular.pth
```

3. Sobel Edge Preprocessing + CNN + Residual Concept

```bash
python test.py -net 2 -weights path/to/checkpoint/100-regular.pth
```

4. Ensemble Network (1+2+3)

```bash
python test.py -net 3 -weights path/to/checkpoint/100-regular.pth -model path/to/checkpoint/100-regular.pth -model2 path/to/checkpoint/100-regular.pth -model3 path/to/checkpoint/100-regular.pth
```

### Tensorboard

```bash
tensorboard --logdir='runs'
```

#### 옵션 정리

**공통 옵션**

- -b : batch size(default value : 32)
- -num_workers : pytorch DataLoader에서 데이터를 로드하는 동안 사용할 서브 프로세스의 수(default value : 8)
- -net : 모델 선택(default value : 0)
  - 0 : CNN 모델
  - 1 : Canny Edge 전처리 + CNN + Residual
  - 2 : Sobel Edge 전처리 + CNN + Residual
  - 3 : Ensemble 모델

  **만약 -net이 3일 경우**

  - -model : Ensemble 모델에서 첫 번째 모델에서 사용할 가중치 파일
  - -model2 : Ensemble 모델에서 두 번째 모델에서 사용할 가중치 파일
  - -model3 : Ensemble 모델에서 세 번째 모델에서 사용할 가중치 파일

**train.py**
- -lr : learning rate(default value : 0.001)

**test.py**

- -weights : 사용할 모델의 가중치 파일

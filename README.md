# DiamondWire

# Model Architecture

![Ensemble](https://github.com/ans2568/DiamondWire/assets/80823431/b9072e02-481b-498e-9988-9ac57507de8c)

# How to Usage

### Train

```bash
python train.py -b 16 -lr 0.001 -num_workers 4
```

### Test

```bash
python test.py -b 16 -lr 0.001 -num_workers 4
```

### Tensorboard

```bash
tensorboard --logdir='runs'
```

#### 옵션 정리

- -b : batch size
- -lr : learning rate
- -num_workers : pytorch DataLoader에서 데이터를 로드하는 동안 사용할 서브 프로세스의 수

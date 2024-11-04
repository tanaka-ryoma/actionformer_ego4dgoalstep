# Requirements

- Linux
- Python 3.5+
- PyTorch 1.11
- TensorBoard
- CUDA 11.0+
- GCC 4.9+
- 1.11 <= Numpy <= 1.23
- PyYaml
- Pandas
- h5py
- joblib

# Compilation

Part of NMS is implemented in C++. The code can be compiled by

```shell
cd ./libs/utils
python setup.py install --user
cd ../..
```

The code should be recompiled every time you update PyTorch.


torch                   1.8.1+cu101
torchaudio              0.8.1
torchvision             0.9.1+cu101

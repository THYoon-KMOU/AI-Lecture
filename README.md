# Step 1. NVIDIA GPU 확인
1. nvidia-smi

# Step 2. CUDA Version 확인
https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications 

1. 본인 GPU에 맞는 Compute Capability 확인(예: Geforce RTX 3060/4090의 경우, 8.6/8.9)
2. Compute Capability에 맞는 CUDA SDK Version(s) 확인(예: Geforce RTX 3060/4090의 경우, 11.1~11.5/11.8)

# Step 3. CUDA Toolkit 설치
https://developer.nvidia.com/cuda-toolkit-archive

1. 본인 CUDA SDK Version에 맞는 CUDA Toolkit 설치

# Step 4. CUDA Version에 맞는 cuDNN 설치
https://developer.nvidia.com/rdp/cudnn-archive

cuDNN(CUDA Deep Neural Network library): 딥러닝 모델 학습을 위한 CUDA 기반의 GPU 가속 Deep-Learning Library
1. 본인 CUDA Version에 맞는 cuDNN Archive 설치(예: Geforce RTX 3060/4090의 경우, CUDA 11.2/11.x)
2. 압축파일 해제 후, 각 파일에 해당되는 경로에 파일 내용을 덮어씀(path: c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\ = )
3. 시스템 환경 변수 편집 -> CUDA_PATH, CUDA_PATH_Vversion이 잘 설정되었는 지 여부 확인
4. cmd 창에 nvcc --version 입력 후, 설치한 버전이 확인되면 성공

# Step 5. Anaconda 가상환경 구축
## 5.1. 가상환경 확인 및 생성

1. conda info --envs
2. conda create -n 가상환경이름 python=파이썬 버전
3. conda activate 가상환경이름
4. conda deactivate

* 파이썬 버전 확인방법: python --version
  
## 5.2. 학습에 필요한 각종 Library 설치

1. 

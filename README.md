# **Document Type Classification | 문서 타입 분류**

## **Team: CV_4조**

**0. Overview**

**Environment**

- Ubuntu 20.04
- python
- Geforce RTX3090 24GB
- wandb

**Requirements**

- albumentations==1.3.1
- ipykernel==6.27.1
- ipython==8.15.0
- ipywidgets==8.1.1
- jupyter==1.0.0
- matplotlib-inline==0.1.6
- numpy==1.26.0
- pandas==2.1.4
- Pillow==9.4.0
- timm==0.9.12

**1. Competiton Info**

**Overview**

- 경진대회 주제
    
    문서 타입 분류를 위한 이미지 분류(17종)
    

**Timeline**

- 2024.04.11 ~ 2024.04.23 19:00

**2. Components**

**Directory**

```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
|   |   └── hyper_tuning.ipynb
├── docs
│   ├── pdf
│   │   └── Upstage AI Lab 2기_CV_04조.pdf
│   └── paper
└── input
    └── data
        ├── train.csv 
        └── pred.csv
        └── submission.csv

```

**3. Data descrption**

**Dataset overview**

데이터는 총 17개 종의 문서로 분류되어 있음. 

1570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측하게 됨. 

현업에서 사용하는 실 데이터를 기반으로 대회를 제작하여 대회와 현업의 갭을 최소화

현업에서 생길 수 있는 여러 문서 상태에 대한 이미지를 구축.

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/5d279e86-ae98-4bb3-8cc1-5eccd741f460)

- **train [폴더]**
    - 1570장의 이미지가 저장되어 있습니다.
- **train.csv [파일]**
    
    ![image](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/832b4982-bd93-4480-936f-3c93a1aee98b.png)
    
    - 1570개의 행으로 이루어져 있습니다. `train/` 폴더에 존재하는 1570개의 이미지에 대한 정답 클래스를 제공합니다.
    - `ID` 학습 샘플의 파일명
    - `target` 학습 샘플의 정답 클래스 번호
- **meta.csv [파일]**
    
    ![image](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/d4b872ca-b669-4166-b146-5ce12af01deb.png)
    
    - 17개의 행으로 이루어져 있습니다.
    - `target` 17개의 클래스 번호입니다.
    - `class_name` 클래스 번호에 대응하는 클래스 이름입니다.
- **test [폴더]**
    - 3140장의 이미지가 저장되어 있습니다.
- **sample_submission.csv [파일]**
    - 3140개의 행으로 이루어져 있습니다.
    - `ID` 평가 샘플의 파일명이 저장되어 있습니다.
    - `target` 예측 결과가 입력될 컬럼입니다. 값이 전부 0으로 저장되어 있습니다.

![image](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/86c6b7ed-f8a4-4909-a614-a8d3bdfc94a7.png)

그 밖에 평가 데이터는 학습 데이터와 달리 랜덤하게 Rotation 및 Flip 등이 되었고 훼손된 이미지들이 존재합니다.

**EDA**

- **EDA**
    - 데이터 분포  ⇒ 특정 클래스에서 데이터 불균형이 있음.
    
    ![image](https://github.com/UpstageAILab2/upstage-cv-classification-cv-04/assets/114049128/9e376d2c-4513-42cb-8a2a-9c488a56559c)
    
    - 기울어진 데이터 존재
    
    ![image](https://github.com/UpstageAILab2/upstage-cv-classification-cv-04/assets/114049128/b098321d-933e-4b2e-944f-82f6babd708a)
    
    - mix up 된 데이터 존재
    
    ![image](https://github.com/UpstageAILab2/upstage-cv-classification-cv-04/assets/114049128/db2af32b-7c87-4e65-8bfc-4aeeab9eabf4)
    
    - 그 외의 다양한 데이터 변형이 가해져 있음.
    
    ![image](https://github.com/UpstageAILab2/upstage-cv-classification-cv-04/assets/114049128/35442c84-e049-498f-9104-6242a1d7e784)
    
    ![image](https://github.com/UpstageAILab2/upstage-cv-classification-cv-04/assets/114049128/1c635f74-1eac-4611-a28e-31301aada926)
    
    ![image](https://github.com/UpstageAILab2/upstage-cv-classification-cv-04/assets/114049128/7dc8a0f2-815e-492b-8be6-e35cc5a71fa4)
    
    - 클래스별 f1스코어 ⇒  class 3, 7,14의 f1스코어가 낮은 것을 확인할 수 있음
    
    ![image](https://github.com/UpstageAILab2/upstage-cv-classification-cv-04/assets/114049128/b9bff79c-166e-4591-9507-1a183adbf6fc)
    
    class 3  confirmation_of_admission_and_discharge
    
    ![image](https://github.com/UpstageAILab2/upstage-cv-classification-cv-04/assets/114049128/703934f7-307a-45e2-b149-8faf5e04a403)
    
    class 7   medical_outpatient_certificate
    
    ![image](https://github.com/UpstageAILab2/upstage-cv-classification-cv-04/assets/114049128/30c15251-ea2d-4049-8fff-3be67c22193d)
    
    class 14   statement_of_opinion
    
    ![image](https://github.com/UpstageAILab2/upstage-cv-classification-cv-04/assets/114049128/e0e4ce26-4f13-4b77-93cd-ecc0aa75e1c5)
    

**Data Processing**

```python
 RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), p=0.5)
 Resize(height=img_size, width=img_size),
 HorizontalFlip(p=0.5),
 VerticalFlip(p=0.5),
 RandomRotate90(p=0.5),
 Rotate(limit=(-35, 35), p=0.5),
 GaussianBlur(blur_limit=(3, 7), p=0.5),
 GaussNoise(always_apply=False, p=0.5, var_limit=(50.0, 200.0), per_channel=True, mean=0.0),
 HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
 RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
 # 클래스 간 차이 부각을 위한 추가 기법
 ImageCompression(quality_lower=60, quality_upper=100, p=0.5),  # 이미지 압축 및 품질 저하
 CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5), # CoarseDropout 추가
 Normalize(mean=[0.57433558, 0.58330406, 0.58818927], std=[0.18964056, 0.18694252, 0.18506919]),
 ToTensorV2()
```

- Normalize의 경우는 train, test 데이터의 평균, 분산을 직접 계산해서 적용
- 추가로 mix up을 적용

**4. Modeling**

**Model description**

- EffecientNet-B4
- ResNet에 **Compound Scaling**을 접목시켜 **Parameter 수 대비** **달성** **가능한** **Accuracy**를 극대화가 가능함.
- 연구자들은 Compound Scaling을 시도하며 총 8개의 모델을 실험하는데, B4가 **Sweet Spot**에 해당함.
- 자체 실험 결과 ResNet 대비 F1 Score 상승함을 보임.
- B5 모델 이상은 필요로 하는 이미지 사이즈가 더욱 커 OOM 문제가 발생.

**Modeling Process**

- Stratified K-Fold Cross Validation

![image](https://github.com/UpstageAILab2/upstage-cv-classification-cv-04/assets/114049128/2cf2179e-48af-4c66-b8f4-023b5261a700)

**5. Result**

**Leaderboard**

![image](https://github.com/UpstageAILab2/upstage-cv-classification-cv-04/assets/114049128/269a331c-2bf3-45ea-800a-3dcaae17d2fc)

- 3위 Score: 0.9383

**Presentation**

- *Insert your presentaion file(pdf) link*

**Reference**

- https://bitrader.tistory.com/226
- [https://arxiv.org/pdf/1710.03740.pdf）。](https://arxiv.org/pdf/1710.03740.pdf%EF%BC%89%E3%80%82)
- 

**Personal Assesments**

- 풍부한 의사소통 기반 역할 분담이 원활히 이루어짐.

# Document Type Classification | 문서 타입 분류

## 1. Competiton Info

### Overview

- 문서 타입 분류 대회는  computer vision domain에서 가장 중요한 태스크인 이미지 분류 대회입니다.

이미지 분류란 주어진 이미지를 여러 클래스 중 하나로 분류하는 작업입니다. 이러한 이미지 분류는 의료, 패션, 보안 등 여러 현업에서 기초적으로 활용되는 태스크입니다. 딥러닝과 컴퓨터 비전 기술의 발전으로 인한 뛰어난 성능을 통해 현업에서 많은 가치를 창출하고 있습니다.

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/b0e34864-01ae-4d65-95fe-5aa96d31606b)


- 본 대회는 결과물 csv 확장자 파일을 제출하게 됩니다.

  input : 9,272개의 아파트 특징 및 거래정보

  output : 9,272개의 input에 대한 예상 아파트 거래금액

### Timeline

-  March 20, 2024 - Start Date
-  april 2, 2024 - Final submission deadline

### Evaluation


![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/01016fd0-bc05-4706-8cb6-d1214b1aae36)


 
- 해당 시점의 매매 실거래가를 예측하는 Regression 대회이며, 평가지표는 RMSE(Root Mean Squared Error)를 사용합니다.

## 2. Components

### Directory

- _Insert your directory structure_

## 3. Data descrption

### Dataset overview

- 데이터 형태:  csv 파일
- 데이터 기간:  train(2007.01.01 ~ 2023.06.30),  test(2023.07.01 ~ 2023.09.26)
- 데이터 개수: train(1118822개), test(9272개)
- 아파트 정보에 대한 변수: 52개( + 거래시점에 대한 변수)
- 서울시 지하철역, 버스 정류장에 대한 데이터 추가

  ![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/c1a5b438-7275-4bca-b18e-e7bc1b687abe)





- 변수별 결측치 비율

1. 좌표 X,Y의 결측치 비율이 높음 =>활용하기 위해서는 데이터를 추가할 필요가 있다.
2. 앞에 ‘K-’가 붙은 변수들은 결측치의 비율이 높다.


![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/37dc9757-5df7-44a0-9d57-c270e4c8c911)


- 아파트의 매매가를 결정하는데에 교통적인 요소가 영향을 줄 수 있기에 추가 데이터로 서울시 지하철역, 서울시 버스정류장의 정보가 주어집니다. 

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/81049731-6d7c-472e-b3bf-5f581d86728e)






### EDA

- ‘target’의 분포 =>  right-skewed

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/3fbad974-e785-4324-8cf3-4e9ca29789c1)


-  ‘층’의 분포 =>  대부분 20층 이내

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/057246fb-a754-4540-ad4a-b6db01cda738)


-  ‘전용면적’의 분포  => 대부분 100㎡ 이내

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/3fcbb084-47fa-4857-9f54-f914cae61960)


- ‘전용면적’과 ‘target’의 분포  => 전용면적이 클수록 가격이 상승

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/9cb22cae-7073-4acf-98d4-1174a2eb7986)

- ‘층’과 ‘target’의 분포

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/a6bd6804-fd00-4712-9c69-b078f6862ee8)


- ‘구’별 실거래가 => 강남,서초,성동,용산구에 높은 거래가

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/826804d2-7e7c-4448-ab79-dcd08c44a7c3)


- ‘구’별 실거래수 => 노원구가 거래수가 제일 많음

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/b4cf1051-1ca6-4dba-a402-b4c83d125370)


- ‘년도’별 실거래가 => 시간이 지날수록 가격 상승, 2020년이후로 가격이 급격하게 상승

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/88c1fa41-9c33-437d-ac15-68dbd09a66e7)


- ‘년도’별 실거래수 => 2015년도에 거래수가 제일 많음

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/53ee2404-5378-40a8-a103-9ebeb8759715)




### Feature engineering

#### ｢결측치 처리｣

- 특정 의미 없는 value np.nan 처리
- 기본적으로 결측치가 90% 이상을 차지하는 피처 → Deletion
- 나머지 피처 → Imputation
  - 연속형 변수: 선형보간 / 범주형 변수: NULL 임의의 범주로 대체

**[본번/부번]**

- 도로명의 경우, 결측치 X
- 도로명을 기준으로 번지 검색 후  Imputation
  - ‘현릉로8길 10-22’ → 특정 하나의 도로명에 대한 본번/부번 결측 확인

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/eb3d2cb3-801a-400b-b252-c2d0d44c49f3)


**[좌표X/좌표Y]**

- 네이버 API를 사용하여 x, y 결측치(경도, 위도) Imputation
- 도로명 주소를 사용하여 x, y 결과값 획득

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/b06387c5-6480-4c85-938b-f0ee96d9d790)


#### ｢컬럼 삭제｣


1. 개념이 일정 부분 겹치는 경우
a. 면적 관련 컬럼 상관관계 확인
  - ‘k-연면적’, ‘k-주거전용면적’, ‘k-관리비부과면적’ 상관계수 높게 기록 
  - k-연면적 제외 나머지 두 칼럼 삭제
    - 연면적: 하나의 건축물의 바닥면적 합계
    - 주택공급면적: 주택전용면적 + 주거공유면적
    - 대부분의 아파트에서 관리비 부과 시, 주택공급면적 기준으로 배분

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/5a8ada58-973d-410b-b5b8-695c84779c18)



 b. 번지/본번/부번/도로명
		- 번지 = 본번 + 부번
		- 번지와 도로명은 지번과 도로명의 주소방식 차이
		- 번지 삭제

2. 학습 효율성
a. 전화번호, 팩스번호
		- 기입방식이 제각각
		- feature importance 낮은 순위
		- 전처리를 통해 기입방식을 통일시키는 것보다, 아예 삭제하는 것이 효율적이라 판단

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/aa33b3dd-d40c-49bc-a73a-98db53a06bea)


#### 외부데이터 활용

- 가계 대출 규모	(3개월 ~ 6개월):	가계 대출 증가는 주택 구매 여력 증가로 이어질 수 있지만, 실제 주택 구매까지는 3개월 ~ 6개월 정도의 시간이 소요될 수 있습니다.


- 한국 기준금리	(6개월 ~ 1년):	금리 변화는 주택 가격에 직접적인 영향을 미치지만, 그 영향은 즉각적으로 나타나지 않고 6개월 ~ 1년 정도의 시차를 두고 나타납니다.


- 미국 기준금리	(6개월 ~ 1년): 	미국 금리 변화는 국내 금리 및 주택 가격에 영향을 미칠 수 있으며, 6개월 ~ 1년 정도의 시차를 두고 나타납니다.


- 본원통화량	(6개월 ~ 1년): 	통화량 증가는 인플레이션으로 이어져 주택 가격 상승으로 이어질 수 있지만, 그 영향은 직접적이지 않고 다른 변수들에 의해 영향을 받으며, 6개월 ~ 1년 정도의 시차를 두고 나타납니다.


- 인허가 실적	(6개월 ~ 1년): 	인허가 증가는 향후 주택 공급 증가를 나타내지만, 실제 주택 공급까지는 6개월 ~1개월의 시차를 두고 나타납니다.


- 아파트 미분양 현황	(3개월 ~ 6개월): 	미분양 주택 증가는 주택 공급 증가로 이어져 주택 가격 하락으로 이어질 수 있으며, 3개월 ~ 6개월 정도의 시차를 두고 나타납니다.

- 월별 아파트 거래량	(1개월 ~ 3개월): 	주택 거래량 증가는 주택 수요 증가를 나타내며, 1개월 ~ 3개월 정도의 시차를 두고 주택 가격에 영향을 미칠 수 있습니다.



#### Feature Engineering_baseline

- 분할 가능 변수
  - 시군구 → 구/동
  - 계약년월 → 계약년/계약월

- 강남 여부

  - 강남/강북 기준으로 
  - 파생변수 생성

- 신축 여부

- 2009년 기준으로 신축 여부 판단

#### 전용면적 구분

**‘전용면적’ 이상치로 판단할 논리적인 근거 부족**
- 대체적으로 전용면적이 클 수록, 집값이 높아지는 경향
- 전용면적이 클수록 집값 분포가 커지는 다른 요인 존재한다 판단 → 전용면적별 파생변수 생성하여 세부적으로 학습할 수 있도록 함 

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/f4cbbb42-74de-4a13-a5b0-69a9f578af50)


**전용면적 별로 feature 생성**

- 아래 기준으로 4개의 피처 생성
- 범주에 해당하는 경우 1, 그렇지 않으면 0

  - 전용면적 <= 60 
  - 60 < 전용면적 <= 85
  - 85 < 전용면적 <= 135
  - 135 < 전용면적
  
#### 역세권 여부


- 네이버 API로 얻은 좌표X, 좌표Y를 기준으로 정의
- 역세권에 대한 정의는 철도(지하철)를 중심으로 3km안에 해당 철도가 있을 경우 True로 간주
- 역세권 여부 포함하지 않는 경우, 3km, 10km 실험 trial

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/482b33e0-df20-4061-b4f2-60728b772a4e)
(https://www.honestfund.kr/blog/money/p2p/station-influence-area)

## 4. Modeling

### Trial. 모델 선택

- 회귀 문제에서 자주 쓰는 Gradient Boosting 계열 LGBM, XGB, CatBoost  모델에 대하여 실험

- default params 기준, CatBoost의 성능이 가장 좋게 나옴

- 다만, CatBoost는 default가 잘 세팅되어있고,
  LGBM은 가장 hyper-params-sensitive한 모델임을 고려

**두 모델을 기준으로 여러가지 실험/앙상블 지속적으로 진행**

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/2630a619-ffb4-41d5-a62a-58958bb4985f)

(Public LB: 위에서 부터 XGB, LGBM, CatBoost)

### Trial. 전용면적 기준 및 모델생성


- Baseline 코드 기준, 기초적인 EDA / 전처리 진행 시, 전용면적이 클 경우에 대하여 모델이 예측 성능이 떨어짐을 확인
  - Idea1) 전용면적이 높은 데이터를 분할 → 따로 모델생성
  - Idea2) 전용면적을 기준으로 하는 피처 생성 
  - **Idea3) 전용면적 기준 피처 생성 + 전용면적별로 모델생성**


- Idea1) 전용면적을 기준으로 모델 생성 진행하였으나, LB 점수가 2 - 3배 가량 오름(성능 하락)
  - 기준: 122(IQR, upper bound) / 100(실평수 30평)


- Idea 2, 3) 전용면적 기준은 앞선 k컬럼의 기준 활용 → 모델 생성 진행
  - 가장 높은 public LB 기록(전용면적별로 모델생성 상세)

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/5c4229d4-3738-45f7-91de-d09f913ea895)

(Public LB: 위에서 부터 100 기준 생성 평균, 100 기준 생성, 122 기준 생성 평균, 122 기준 생성)


### Trial. 외부 금융 데이터


- 금융 데이터 전체 사용할 경우, 10만대의 RMSE LB 기록
- 금융 데이터에 대한 전처리는 필수적
  - Idea1) 금융 데이터 일부 채용
  - Idea2) 금융 데이터 시차 적용

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/79b650bc-0ca1-4689-be21-f0e1a3bb46dc)

(Public LB)

- 상관계수 활용하여 target, 계약년도와 상관계수가 높은 피처 선택하여 모델 진행 하였을 경우, Feature importance는 높게 찍혔으나, 여전히 LB가 10만대 
    - 금융 피처별 상관성이 높은 피처 그룹중 하나씩 선택
    - ‘기준금리’, ‘ 가계대출금액’


![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/e2a6afc2-9790-4e06-ab85-8924d196523e)


![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/81ffbad3-d3d1-42db-9344-347838b046a8)



### Trial. 좌표X, Y 데이터 및 역세권 여부 기준

- 좌표 X, Y
  - Idea1) 결측치 네이버 API로 대체
  - Idea2) X,Y 제거 시
  - Idea3) 역세권 여부 피처 추가

- 결론적으로는 x,y 피처의 결측치를 채우고 모델을 학습시켰을 시, 성능이 오히려 큰 폭으로 하락
- X,Y 제거 했을 경우, 성능 대폭 향상됨을 확인할 수 있었음
- 역세권 여부 피처 3, 10km 기준으로 추가했을 경우, 역세권 여부 피처에 대한 Importance와, 성능 향상이 유의미하지 않았음을 확인

**좌표 X,Y 제거한 모델이 가장 성능이 좋은 것으로 결과**

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/693e0dda-32c6-4ec7-8d0d-978dd6991852)

(Public LB: 좌표데이터제거)


### Trial. 전용면적별로 모델생성


- 전용면적 범위를 기준으로 4개의 데이터로 분할 후, 각 데이터마다 모델 학습 및 추론 후 모델 생성
  (A B C D 모델로 통칭)

- 파라미터 최적화 되지 않았음에도 성능 향상

- A B C D 4개로 분할된 각 모델마다 Optuna로 hyperparameter tuning 최적화

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/fd66c2c9-6c93-44e5-9917-2c49eb658248)    >>    ![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/b6f5508b-0e76-45e5-aa16-5fe158b02ff8)

기존 모델 (튜닝 최적화)          >>           전용면적 모델(튜닝 최적화 X)


![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/50c6d5e7-208d-4c2a-960d-5f7a5f5ce4d8)

(구간별 데이터 분포)

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/cc2e9ad7-b924-4511-91be-72aaf6e1ed04)

(모델별 RMSE )


- A B 모델은 데이터 비중이 높음
- C D 모델의 RMSE가 높음

- 각 데이터의 특성에 따라 맞춤 튜닝

- 전용면적 별로 모델 빌드 후 
  Public LB 8위 >> Public LB 5위

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/b6f5508b-0e76-45e5-aa16-5fe158b02ff8)    >>   ![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/e3fef565-561a-4f55-9f20-a2c691be453d)

전용면적 모델(튜닝 최적화 X)       >>       전용면적 모델(튜닝 최적화)



## 5. Result

### Leader Board

![image](https://github.com/UpstageAILab2/upstage-ml-regression-4/assets/114049128/36bae580-5612-4a9c-8c12-0f2e4941608c)


### Presentation

- https://github.com/UpstageAILab2/upstage-ml-regression-4/blob/f82b22bd86bf28f712d618623c46737d33a5035e/docs/pdf/4%EC%A1%B0_%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C.pdf


### 진행 소감

1. 외부데이터 적용의 어려움

	이유 : 간단한 날짜에 대한 데이터도 저장된 형태도 다르고 데이터셋의 구조가 다르기 떄문에 합치는 과정에서 많은 어려움이 있었다.
	
 	향후 계획 : 실제 대회를 해봐야 알 수 있는 문제상황을 마주쳤다는 생각이 들고 그렇기에 유익한 경험이라는 생각이 든다.

3. Overfitting 판단의 어려움

	이유 : 학습시킨 모델의 성능을 평가할 때에 validation의 RMSE 와 실제 LB의 validation에 대한 차이가 너무 커서, test 데이터셋을 평가하는데 있어 어려움이 있었다.
	
 	향후 계획 : 시간 관계상 여러 CV 방법을 적용해보지 못하였기 때문에 다양한 CV 방법을 시도해보면 더 좋을 것 같다.

5. 모델 앙상블의 새로운 접근
	
	이유 : LGBM, CatBoost 등 알고리즘 모델 단위로 앙상블 하는 것을 일반적으로 생각했었는데 하나의 컬럼을 기준으로 분할 학습, 앙상블 하는 idea를 멘토님 덕에 알게되었고, 직접 앙상블 코드를 구축하고 끝내 성능 향상을 이뤄 뜻깊은 경험이 되었다.
	
 	향후 계획 : LGBM 모델로만 앙상블 했었는데, 다른 모델/분할 기준으로 더 효율적인 앙상블 방법을 고민해볼 것.





## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_

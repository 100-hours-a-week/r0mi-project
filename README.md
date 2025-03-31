# 🍅 Tomato Leaf Disease Classification with Pretrained CNNs

본 프로젝트는 토마토 잎의 질병 이미지를 분류하기 위한 이미지 분류 모델을 구축하고,  
ResNet34, EfficientNet-B0, DenseNet121의 세 가지 사전 훈련 CNN 모델을 비교 분석한 실험 결과를 정리한 것입니다.

---

## 📌 프로젝트 개요

- **목적**: 토마토 잎 이미지 데이터를 활용해 질병 분류 모델을 학습하고, 사전 훈련 CNN 모델 간의 성능을 비교
- **데이터셋**: [Tomato Disease - Multiple Sources (Kaggle)](https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources)
- **클래스 수**: 총 11개 (정상 1종, 질병 10종)
- **이미지 수**: 총 32,535장 (Train 25,851장 / Valid 6,684장 / Test 별도 분할)

---

## 🧪 실험 구성

| 실험 조건 | 내용 |
|-----------|------|
| 모델 | ResNet34, EfficientNet-B0, DenseNet121 |
| 실험 버전 | 기본 학습(baseline), StepLR 적용, EarlyStopping 적용 |
| 데이터 전처리 | Resize(224), Normalize, RandomHorizontalFlip (train only) |
| 손실 함수 | CrossEntropyLoss |
| Optimizer | Adam |
| 초기 학습률 | 0.0001 |
| 배치 크기 | 32 |
| 에포크 | 25 epochs |

---

## 📊 주요 결과

모델별 Test Accuracy 비교:

| 모델 | 버전 | Test Accuracy |
|------|------|----------------|
| ResNet34 | baseline | 0.9812 |
| ResNet34 | StepLR | 0.9716 |
| ResNet34 | EarlyStopping | 0.9812 |
| EfficientNet-B0 | baseline | 0.9817 |
| EfficientNet-B0 | StepLR | 0.9913 |
| EfficientNet-B0 | EarlyStopping | 0.9830 |
| DenseNet121 | baseline | 0.9716 |
| DenseNet121 | StepLR | 0.9860 |
| DenseNet121 | EarlyStopping | 0.9716 |

📌 **가장 높은 성능**: EfficientNet-B0 + StepLR (0.9913)

---

## ✅ 결론 요약

- 사전 훈련 CNN 모델은 토마토 질병 분류에 효과적이며, 전이 학습 전략이 성능 향상에 기여함
- EfficientNet-B0가 전체적으로 가장 안정적이고 우수한 성능을 보임
- StepLR은 일부 모델에서 성능 향상을 유도했으며, EarlyStopping은 과적합을 방지하는 데 기여함
- 데이터셋에 약간의 클래스 불균형이 있었지만 전체적인 성능에는 큰 영향을 미치지 않음

---

## 🗃️ DB 연동

- 학습 결과(test loss, test accuracy)를 SQLite에 저장
- 테이블: `training_logs`
- 필드: id, model_name, version, epoch, start_time, end_time, test_loss, test_accuracy

---

## 부록
- [▶️ Google Colab - 기본 학습(baseline)](https://colab.research.google.com/drive/10bLWW1Jk0TLRU0FuS1UgfASY5-CxEn8q?usp=sharing)
- [▶️ Google Colab - StepLR 적용](https://colab.research.google.com/drive/1zShPnD1pH7jtz5UjeOz5gXoJA8S2tswU?usp=sharing)
- [▶️ Google Colab - Early Stopping 적용](https://colab.research.google.com/drive/1aiYD23stLwXgdI03OcrwB4PTO1e3TCfc?usp=sharing)







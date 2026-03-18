# PhosOpt 전체 코드 구조 및 설계 설명서

## 1. 프로젝트 개요

피질 시각 보철(cortical visual prosthesis)의 **최적 임플란트 배치**를 찾는 프로젝트입니다.

핵심 질문: "임플란트를 뇌에 한 번 삽입할 때, 어떤 각도/위치로 놓아야
다양한 이미지(글자 등)를 전극 활성화만으로 가장 잘 표현할 수 있는가?"

### 출력

- **4개 공유 파라미터** (모든 이미지에 공통, 임플란트 배치)
  - alpha: 회전 각도 (-90 ~ 90도)
  - beta: 기울기 각도 (-15 ~ 110도)
  - offset_from_base: 베이스 오프셋 (0 ~ 40mm)
  - shank_length: 샹크 길이 (10 ~ 40mm)
- **1000개 전극 활성화** (이미지마다 다름)

---

## 2. 디렉토리 구조

```
phosopt/
├── environment.yml          # Conda 환경 정의
├── README.md                # 프로젝트 안내
│
├── basecode/                # 참조용 원본 코드 (직접 실행하지 않음)
│   ├── model.py             #   E2E_Encoder/Decoder 원본
│   ├── ninimplant.py        #   임플란트 좌표 변환 유틸
│   ├── electphos.py         #   전극-phosphene 매핑
│   ├── lossfunc.py          #   원본 손실 함수
│   ├── visualsectors.py     #   시각 영역 유틸
│   └── bayesianopt_V1.ipynb #   베이지안 최적화 노트북
│
├── code/                    # 실제 학습/실행 코드
│   ├── train.py             #   [진입점] 학습 스크립트
│   ├── trainer.py           #   학습/평가 루프
│   ├── dataset.py           #   데이터셋 로딩/분할
│   │
│   ├── models/              #   신경망 모델
│   │   ├── __init__.py
│   │   ├── encoder.py       #     E2E 스타일 CNN 인코더
│   │   ├── parameter_head.py#     ElectrodeHead, ContinuousHead, ParameterBounds
│   │   └── inverse_model.py #     InverseModel (공유 파라미터 + 인코더)
│   │
│   ├── simulator/           #   물리 시뮬레이터
│   │   ├── __init__.py
│   │   ├── physics_forward.py      # NumPy 기반 forward 시뮬레이션
│   │   ├── physics_forward_torch.py# PyTorch 미분가능 시뮬레이터
│   │   └── simulator_wrapper.py    # 시뮬레이터 래퍼/어댑터
│   │
│   ├── loss/                #   손실 함수
│   │   ├── __init__.py
│   │   └── losses.py        #     MSE, Dice, Sparsity, 등 조합
│   │
│   ├── diagnose.py          #   [도구] 파이프라인 진단
│   ├── diag_sim_upper_bound.py #  [도구] 시뮬레이터 상한 테스트
│   └── analyze_letters_structure.py # [도구] 데이터 구조 분석
│
└── docs/                    # 문서
    ├── current_issues.md
    ├── design_shared_params_ko.md  # (이 문서)
    └── design_shared_params_en.md
```

---

## 3. 핵심 설계: 공유 임플란트 파라미터

### 이전 구조 (per-image)

```
이미지 -> Encoder -> latent -> ContinuousHead -> 4 params  (이미지마다 다름)
                            -> ElectrodeHead  -> 1000 logits (이미지마다 다름)
```

**문제**: 임플란트는 한 번 삽입하면 움직일 수 없는데,
이미지마다 다른 임플란트 배치를 예측하는 것은 물리적으로 불가능합니다.

### 현재 구조 (shared params)

```
+------------------------------------------+
|              InverseModel                |
|                                          |
|  _shared_params_raw  (nn.Parameter [4])  |  <- 전체 공유, 학습됨
|       |                                  |
|       v sigmoid + bounds scaling         |
|  shared_params [4]                       |  <- alpha, beta, offset, shank
|       | .unsqueeze(0).expand(B, -1)      |
|       v                                  |
|  params [B, 4]  (모든 행이 동일)           |
|                                          |
|  image [B,1,256,256]                     |
|       |                                  |
|       v                                  |
|  Encoder (E2E 스타일 CNN)                 |
|       |                                  |
|       v                                  |
|  latent [B, 256]                         |
|       |                                  |
|       v                                  |
|  ElectrodeHead (MLP)                     |
|       |                                  |
|       v                                  |
|  electrode_logits [B, 1000]              |  <- 이미지별
|                                          |
|  return (params, electrode_logits)       |
+------------------------------------------+
```

---

## 4. 모듈별 역할 상세

### 4.1 `train.py` — 진입점

- CLI 인자 파싱 (`--maps-npz`, `--retinotopy-dir`, `--lr`, `--shared-params-lr` 등)
- 데이터셋 로딩 및 train/val/test 분할
- DataLoader 생성
- 모델/시뮬레이터/손실/학습 설정 구성
- `train_inverse_model()` 호출
- 평가 (역모델, 랜덤 베이스라인, 4-param 베이스라인)
- 모델 가중치 및 리포트 JSON 저장

### 4.2 `trainer.py` — 학습/평가 루프

- `TrainConfig`: 학습 하이퍼파라미터 (epochs, lr, shared_params_lr 등)
- `train_inverse_model()`:
  - **두 개의 optimizer param group**: 네트워크(lr=1e-3) / 공유 파라미터(lr=1e-2)
  - 매 배치: forward -> loss -> backward -> step
  - 매 epoch 첫 배치에서 진단 출력 (shared params 현재 값 포함)
  - epoch 끝에서 validation 평가 + scheduler step
- `evaluate_inverse_model()`: MSE, Dice, Hellinger 메트릭
- `evaluate_random_baseline()`: 랜덤 파라미터/전극으로 비교
- `evaluate_four_param_baseline()`: 공유 파라미터만 사용 + 전극 전부 ON

### 4.3 `models/inverse_model.py` — 핵심 모델

```python
class InverseModel(nn.Module):
    _shared_params_raw: nn.Parameter  # [4], raw (pre-sigmoid)
    encoder: Encoder                   # image -> latent
    electrode_head: ElectrodeHead      # latent -> 1000 logits
```

- `shared_params` 프로퍼티: `sigmoid(raw) * (high-low) + low`로 물리적 범위 보장
- `forward(x)`: 공유 params를 배치 크기로 확장 + 이미지별 전극 logit 예측

### 4.4 `models/encoder.py` — CNN 인코더

basecode의 `E2E_Encoder`를 256x256 입력에 맞게 개조:

```
256x256 -> Conv(16) -> Conv(32)+Pool -> Conv(64)+Pool -> Conv(128)+Pool
        -> ResBlock x4 -> Conv(64) -> Conv(1) -> Flatten(32x32=1024)
        -> Linear(1024, latent_dim=256)
```

- LeakyReLU 사용 (sparse phosphene map에서 "죽는 뉴런" 방지)
- MaxPool 기반 다운샘플링 (E2E 원본 방식)

### 4.5 `models/parameter_head.py` — 출력 헤드

- `ParameterBounds`: 4개 파라미터의 물리적 범위 정의
- `ElectrodeHead`: latent(256) -> MLP -> 1000 logits (현재 사용됨)
- `ContinuousHead`: latent -> MLP -> 4 params (현재 사용 안 됨, 호환성 유지)
- `ParameterHead`: ContinuousHead + ElectrodeHead 래퍼 (현재 사용 안 됨)

### 4.6 `simulator/physics_forward_torch.py` — 미분가능 시뮬레이터

```
(params [B,4], electrode_logits [B,1000])
    -> sigmoid(logits) = electrode_prob
    -> implant geometry from params
    -> retinotopy mapping (PRF)
    -> weighted phosphene rendering
    -> phosphene map [B,1,256,256]
```

- `DifferentiableSimulator`: 전체 과정이 PyTorch autograd와 호환
- retinotopy mgz 파일에서 V1 voxel 정보 로드
- (alpha, beta) 그리드에 대한 표면 거리를 사전 계산

### 4.7 `loss/losses.py` — 손실 함수

`build_losses()` 가 반환하는 총 손실:

```
total = recon_weight * MSE(recon, target)
      + dice_weight * SoftDice(recon, target)
      + warmup * sparsity_weight * mean(sigmoid(electrode_logits))
      + warmup * param_prior_weight * param_prior  (공유 파라미터에서는 항상 0)
      + warmup * invalid_region_weight * invalid_region_penalty
```

- `warmup`: 처음 N epoch 동안 정규화 항을 점진적으로 켬
- `sparsity`: 전극 활성화를 희소하게 유도
- `invalid_region`: 유효하지 않은 전극 활성화 페널티

### 4.8 `dataset.py` — 데이터셋

- `PhospheneDataset`: `.npy` 디렉토리 또는 `.npz`에서 target map 로드
- `normalize_target_map()`: 맵을 [0,1]로 정규화
- `make_splits()`: 비율 기반 train/val/test 분할
- `load_letters_phosphene_splits()`: EMNIST letters 전용 로더

---

## 5. 데이터 흐름 (학습 시)

```
1. 데이터 로드
   emnist_letters_phosphenes.npz -> PhospheneDataset -> DataLoader

2. Forward Pass
   target map [B,1,256,256]
       |
       v
   InverseModel
       |-- shared_params [4] -> expand -> [B,4]  (임플란트 배치, 전체 공유)
       |-- Encoder(target) -> latent [B,256]
       |-- ElectrodeHead(latent) -> electrode_logits [B,1000]  (이미지별)
       |
       v
   DifferentiableSimulator(params, electrode_logits)
       |
       v
   predicted map [B,1,256,256]

3. Loss 계산
   MSE(predicted, target) + Dice + Sparsity + ...

4. Backward + Optimizer Step
   - _shared_params_raw: lr=1e-2 (4개 값, 전체 데이터셋에 걸쳐 수렴)
   - Encoder + ElectrodeHead: lr=1e-3

5. 반복 (epochs x batches)
   -> shared_params: 모든 이미지를 통틀어 최적인 임플란트 배치로 수렴
   -> Encoder/Head: 각 이미지에 맞는 전극 패턴을 예측하는 법을 학습
```

---

## 6. 학습 결과물

### 모델 파일

`<save-dir>/<subject-id>_inverse_model.pt`

```python
model = InverseModel(in_channels=1, latent_dim=256, electrode_dim=1000)
model.load_state_dict(torch.load("s100610_inverse_model.pt"))

# 최적 임플란트 배치 (학습된 공유 파라미터)
print(model.shared_params)  # [alpha, beta, offset, shank]

# 특정 이미지에 대한 전극 활성화
with torch.no_grad():
    params, electrode_logits = model(target_map)
    electrode_prob = torch.sigmoid(electrode_logits)  # [0,1] 연속값
    electrode_onoff = (electrode_prob > 0.5).int()     # 이진 on/off
```

### 리포트 JSON

`<save-dir>/<subject-id>_report.json`

```json
{
  "subject_id": "s100610",
  "learned_implant_params": {
    "alpha": 12.3,
    "beta": 65.7,
    "offset_from_base": 18.2,
    "shank_length": 28.4
  },
  "test_metrics": { "mse": 0.0012, "dice": 0.87, "hellinger": 0.05 },
  "baselines": {
    "random": { "mse": 0.15 },
    "four_params_only": { "mse": 0.08 }
  },
  "config": { "loss": {...}, "train": {...} }
}
```

---

## 7. 실행 방법

```bash
conda activate phosopt

python code/train.py \
  --maps-npz data/letters/emnist_letters_phosphenes.npz \
  --retinotopy-dir data/fmri/100610 \
  --hemisphere LH \
  --subject-id s100610 \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --shared-params-lr 1e-2 \
  --save-dir data/output/inverse_training
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--maps-npz` | (필수) | phosphene map .npz 파일 경로 |
| `--retinotopy-dir` | (필수) | retinotopy mgz 파일 디렉토리 |
| `--hemisphere` | LH | 반구 선택 |
| `--lr` | 1e-3 | Encoder/ElectrodeHead 학습률 |
| `--shared-params-lr` | 1e-2 | 공유 임플란트 파라미터 학습률 |
| `--epochs` | 50 | 학습 epoch 수 |
| `--batch-size` | 8 | 배치 크기 |

---

## 8. 설계 근거

| 결정 | 이유 |
|------|------|
| 4 params를 `nn.Parameter`로 | 물리적으로 임플란트는 한 번 고정됨. `state_dict`에 자동 포함 |
| ContinuousHead 제거 | 이미지별 예측이 아닌 전체 공유이므로 MLP가 불필요 |
| sigmoid + bounds | 학습 중 물리적 범위를 항상 보장 |
| 별도 학습률 | 4개 값은 빠르게, 수만 파라미터 네트워크는 안정적으로 |
| weight_decay=0 (shared) | sigmoid이 이미 범위를 제한하므로 추가 규제 불필요 |
| E2E 스타일 인코더 | basecode의 검증된 구조를 256x256에 맞게 확장 |
| LeakyReLU | sparse한 phosphene map에서 "죽는 뉴런" 문제 방지 |
| Flatten (not AvgPool) | 공간 정보를 최대한 보존 (E2E 원본 방식) |

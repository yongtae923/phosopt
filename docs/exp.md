# 논문화 전략

2026-03-14

# PhosOpt Experimental Design

본 연구는 **phosphene-based cortical visual prosthesis stimulation optimization** 문제에서 제안 방법 **PhosOpt**를 기존 방법들과 비교 평가한다.

연구 질문은 다음 세 가지이다.

1. **Instance-specific optimization 성능**
2. **Shared-parameter generalization 성능**
3. **새 target에 대한 adaptation 효율**

---

# 1. Evaluation Metrics

vimplant 연구에서 사용된 지표를 그대로 사용한다.

**Reconstruction metrics**

- DC (Dice coefficient)
- Yield metric
- HD (Hellinger distance)

**Composite score (vimplant)**

[S = DC + 0.1Y - HD]
[Loss = 2 - S]

**Efficiency metrics**

- active electrode count
- simulator calls
- wall-clock time

효율성은 다음 두 단계로 분리하여 기록한다.

- **training cost** (CNN training)
- **solving cost** (target reconstruction)

---

# 2. Baselines

다음 다섯 가지 방법을 비교한다.

1. **Bayesian optimization (vimplant)**
2. **All-on optimization** (모든 electrode 활성화)
3. **Random subset optimization**
4. **Heuristic subset optimization**
    - center-prior
    - intensity-prior
5. **PhosOpt (proposed)**

---

# 3. Benchmarks

## 3.1 Per-target benchmark

각 target map을 **개별적으로 최적화**하여 성능 비교.

비교 방법

- Bayesian
- all-on
- random subset
- heuristic subset
- phosopt-per-target

---

## 3.2 Generalized benchmark

하나의 **shared parameter**가 여러 target map에서 얼마나 일반화되는지 평가.

비교 방법

- all-on shared
- random shared
- heuristic shared
- phosopt

(Bayesian 제외)

---

## 3.3 Adaptation benchmark

새 target 등장 시 **적응 속도** 비교.

비교 방법

- Bayesian from scratch
- phosopt fine-tune
- phosopt zero-shot

---

# 4. Dataset

Dynaphos simulator를 사용하여 다음 target map을 생성한다.

- **single blob × 5**
- **multiple blobs × 5**
- **arc shapes × 5**
- **MNIST letters × 5**

---

# 5. Optimization Variables

모든 방법은 동일한 parameter space에서 최적화를 수행한다.

[\theta = {p_1, p_2, p_3, p_4, m_1 … m_N}]

- stimulation parameters (p_1,p_2,p_3,p_4) = alpha, beta, offset_from_base, shank_length
- electrode mask (m_i ∈ {0,1}) x 1000

---

# 6. Simulator

모든 방법은 동일한 phosphene simulator를 사용한다.

```
predicted_map = dynaphos(θ)
```

---

# 7. Budget (Fair Comparison)

| Benchmark | Method group | Preparation cost | Solve cost | Early stopping | Seeds |
| --- | --- | --- | --- | --- | --- |
| Per-target | Bayesian / all-on / random / heuristic / phosopt-per-target | 없음 | max 300 simulator calls, max 20 min | 30 calls 동안 개선 < 1e-4 | 5 |
| Generalized | all-on shared / random shared / heuristic shared / phosopt | max 100 epochs, patience 10 | test 시 추가 optimization 없음 | val score 기준 early stop | 5 |
| Adaptation | Bayesian from scratch | 없음 | max 100 calls, max 10 min | 20 calls 동안 개선 < 1e-4 | test target별 |
| Adaptation | fine-tune phosopt | shared model training 완료 상태 | max 100 calls, max 10 min | 20 calls 동안 개선 < 1e-4 | test target별 |
| Adaptation | zero-shot phosopt | shared model training 완료 상태 | 0 calls | 없음 | test target별 |

Early stopping

- score 개선 < 1e−4

Seeds

- 5 seeds

---

# 8. Logging

모든 실험은 동일한 로그 형식을 사용한다.

**Metadata**

```
experiment_id
benchmark_type
method
seed
target_id
```

**Results**

```
score
DC
Y
HD   (Hellinger distance)
active_electrode_count
simulator_calls
wall_clock_time
```

---

# 9. Success Criteria

target reconstruction 성공 기준

```
score ≥ predefined threshold
```

---

# 10. Experimental Pipeline

```
dataset
↓
benchmark selection
↓
method
↓
optimization
↓
dynaphos simulator
↓
metric evaluation
↓
logging
```

---

# 11. Additional Validation

추가 biological validation과 다음 실험을 수행한다.

- **NEURON simulation**
- **DeepRetinotopy constraint evaluation**
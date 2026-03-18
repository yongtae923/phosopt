# Plan / exp.md vs Codebase Audit Report

점검일: 2026-03-15  
기준: `.cursor/plans/phosopt_experiment_restructuring_2a3bda41.plan.md`, `docs/exp.md`

---

## 모든 항목 수정 완료 (2026-03-15)

아래 6건의 불일치가 발견되어 전부 수정 완료되었습니다.

| # | 구분 | 항목 | 수정 내역 |
|---|------|------|-----------|
| 1 | 코드 | Generalized shared 3종 미구현 | `generalized.py`에 `_make_shared_mask`, `_train_shared_baseline`, `_evaluate_shared_on_targets` 추가. all_on_shared/random_shared/heuristic_shared 별도 학습·평가 경로 구현 |
| 2 | 코드 | Adaptation이 test split 미사용 | `experiment.py`의 `run_adaptation()`에서 `targets.split` 읽어 `_split_targets`로 test만 추출하도록 수정 |
| 3 | 코드 | Adaptation Bayesian에 data_dir/hemisphere 미전달 | `adaptation.py`의 `_bayesian_from_scratch`와 `run_adaptation_benchmark`에 data_dir/hemisphere 인자 추가, `experiment.py`에서 전달 |
| 4 | 문서 | Plan에서 HD를 Hausdorff로 기술 | Plan 문서의 hausdorff_distance → hellinger_distance, directed_hausdorff 설명을 Hellinger 공식으로 교체 |
| 5 | 코드 | analyze에 y, loss 미포함 | `analyze.py`의 `summary_table` 메트릭 목록에 `y`, `loss` 추가 |
| 6 | 문서 | exp.md p3,p4 이름 불일치 | exp.md Section 5의 `length, depth` → `offset_from_base, shank_length`으로 basecode 용어 통일 |

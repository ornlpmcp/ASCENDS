# ASCENDS Roadmap

> 마지막 업데이트: 2026-04-11 (v0.4.0 기준)

---

## v0.4.x — 안정성 & UX 개선

- [ ] **로딩 스피너** — Train / SHAP / Correlation 실행 시 진행 표시
- [ ] **SHAP 다중 클래스 수정** — `shap_values` 리스트 반환 시 절댓값 평균으로 통합 (`explain.py:save_default_shap_plot`)
- [ ] **feature alignment 경고** — 예측 시 학습과 다른 feature가 있으면 유저에게 경고 출력 (`data.py:align_to_features`)
- [ ] **`except Exception: pass` 정리** — 광범위한 예외 처리를 `logger.warning()`으로 교체
- [ ] **Run Report** — Save Model 시 `runs/<name>/report.html` 자동 생성, ML Models 패널에 Report 버튼 추가
  - 포함 내용: metrics, parity/confusion 플롯, SHAP importance, 규칙 기반 해석 텍스트
  - 해석 모듈: `ascends/core/interpret.py` — R², MAE, 과적합, F1, 클래스 불균형 등 시나리오 기반

---

## v0.5.0 — 아키텍처 정리

- [ ] **core 중복 제거** — `ascends_server.py`의 `_compute_correlations()` 제거하고 `ascends/core/correlation.py` import로 통일
- [ ] **`task` 표현 통일** — 서버/CLI의 `"r"`/`"c"` → 진입점에서 `canonicalize_task()` 일관 적용
- [ ] **`ascends_server.py` 분리** — correlation / train / predict 라우터를 별도 파일로 분리

---

## v0.6.0 — 모델 해석 & 데이터 진단

- [ ] **데이터 품질 자동 진단** — 결측치 비율, 이상치, 클래스 불균형 감지 + 경고
- [ ] **Baseline 비교** — 더미 모델(평균 예측) 대비 성능 표시
- [ ] **모델 비교 뷰** — runs/ 디렉토리 내 여러 모델 나란히 비교
- [ ] **하이퍼파라미터 튜닝 구현** — 현재 placeholder. Optuna 연동 고려
- [ ] **윈도우 패키징 개선** — bat 스크립트 및 번들 워크플로 안정화

---

## v0.7.0 — 워크플로 확장

- [ ] **전처리 옵션 GUI** — 스케일링, 인코딩 방식 선택
- [ ] **시계열 지원 강화** — time-based split 이미 있음, GUI에서 노출
- [ ] **Classification CLI `--with-proba`** — `predict_proba` 지원 시 확률 컬럼 출력
- [ ] **LLM 해석 옵션** — Claude API 연동으로 데이터 맥락 반영한 해석 (선택적, 오프라인 fallback 유지)
- [ ] **Frontend 개선** — TypeScript + Tailwind 단계적 전환 (Next.js 제외)

---

## 완료

- [x] 중복 matplotlib import 제거 `v0.4.0 · 2026-04-11`
- [x] 파일 업로드 50MB 크기 제한 `v0.4.0 · 2026-04-11`
- [x] Path traversal 취약점 수정 (`/predict/download`) `v0.4.0 · 2026-04-11`
- [x] 중복 top_k 파싱 블록 제거 (corr 설정 덮어쓰기 버그) `v0.4.0 · 2026-04-11`
- [x] return 이후 도달 불가 코드 제거 `v0.4.0 · 2026-04-11`
- [x] 중복 `_unique_preserve()` 함수 제거 `v0.4.0 · 2026-04-11`
- [x] `_unique_run_name` TOCTOU 경쟁 조건 수정 `v0.4.0 · 2026-04-11`
- [x] `LAST_TRAIN` 캐시 최대 20개 제한 `v0.4.0 · 2026-04-11`
- [x] SHAP dual-view (ASCENDS / default beeswarm) 추가 `v0.4.0 · 2026-04-11`
- [x] Classification GUI 지원 `v0.4.0 · 2026-04-11`

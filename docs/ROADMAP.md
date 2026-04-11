# ASCENDS Roadmap

> 마지막 업데이트: 2026-04-11 (v0.4.0 기준)

---

## v0.4.x — 안정성 & UX 개선

- [ ] **로딩 스피너** — Train / SHAP / Correlation 실행 시 진행 표시 (빠름)
- [ ] **SHAP 다중 클래스 수정** — `shap_values` 리스트 반환 시 절댓값 평균으로 통합 (`explain.py:save_default_shap_plot`)
- [ ] **feature alignment 경고** — 예측 시 학습과 다른 feature가 있으면 유저에게 경고 출력 (`data.py:align_to_features`)
- [ ] **`except Exception: pass` 정리** — 20곳 이상의 광범위한 예외 처리를 `logger.warning()`으로 교체

---

## v0.5.0 — 아키텍처 정리

- [ ] **core 중복 제거** — `ascends_server.py`의 `_compute_correlations()` 제거하고 `ascends/core/correlation.py` import로 통일 (2단계: core 인터페이스 정비 → 서버 교체)
- [ ] **`task` 표현 통일** — 서버/CLI의 `"r"`/`"c"` → 진입점에서 `canonicalize_task()` 일관 적용
- [ ] **`ascends_server.py` 분리** — correlation / train / predict 라우터를 별도 파일로 분리

---

## v0.6.0 — 기능 확장

- [ ] **하이퍼파라미터 튜닝 구현** — 현재 placeholder (`tune_trials` 파라미터 미사용). Optuna 연동 고려
- [ ] **윈도우 패키징 개선** — bat 스크립트 및 번들 워크플로 안정화 (`docs/windows_handoff.md` 참고)

---

## 완료

- [x] 중복 matplotlib import 제거
- [x] 파일 업로드 50MB 크기 제한
- [x] Path traversal 취약점 수정 (`/predict/download`)
- [x] 중복 top_k 파싱 블록 제거 (corr 설정 덮어쓰기 버그)
- [x] return 이후 도달 불가 코드 제거
- [x] 중복 `_unique_preserve()` 함수 제거
- [x] `_unique_run_name` TOCTOU 경쟁 조건 수정
- [x] `LAST_TRAIN` 캐시 최대 20개 제한
- [x] SHAP dual-view (ASCENDS / default beeswarm) 추가
- [x] Classification GUI 지원

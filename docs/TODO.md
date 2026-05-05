# TODO

- [ ] 설치 단순화: macOS/Linux/Windows 초보자용 빠른 설치 가이드 + 원클릭 스크립트 정리(의존성 안내 포함).
- [ ] Portable 배포 고도화: Windows 완전 self-contained bundle과 macOS/Linux bundle 기대사항을 구분해 배포 절차 문서화.
- [ ] GUI/domain 경계 정리: `gui_correlation_routes.py`의 correlation 계산 로직을 core로 승격해 CLI와 공유.
- [ ] Neural Network 옵션 추가 검토: `MLPRegressor/MLPClassifier`를 기본모델이 아닌 선택형으로 도입하고 성능/안정성 비교.
- [ ] Hyperparameter tuning 도입: Quick/확장 검색 전략 설계 + Optuna(advanced) 검토.
- [ ] 번들 용량 최적화: `dist` 중복 산출물 정리 규칙 + 용량 목표치(압축본 기준) 설정.
- [ ] Classification CLI 예측 옵션: `--with-proba` 추가(모델이 `predict_proba` 지원 시 확률 컬럼 출력).
- [ ] UI refresh pass: Correlation/Train/Predict 전체 시각 개선(레이아웃/타이포/간격/컴포넌트 일관성).
- [ ] Frontend 개선 PoC 확장: FastAPI + TypeScript + Tailwind 경로를 기존 UI에 단계적으로 적용(Next.js 제외).

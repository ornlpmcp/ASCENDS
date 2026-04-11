# ASCENDS Code Review Notes

> 분석일: 2026-04-11 (최신 커밋 1a2a01c 기준)
> 대상 파일: `ascends_server.py`, `ascends/core/explain.py`, `ascends/core/train.py`, `ascends/core/data.py`

---

## 크리티컬 (즉시 수정 필요)

### 1. 중복 matplotlib import (ascends_server.py:38, 49)
`matplotlib`와 `matplotlib.use("Agg")`가 두 번 import됨.
→ 49-50번 줄 삭제.

### 2. 중복 top_k 파싱 블록 (ascends_server.py:1011, 1042)
top_k를 파싱하고 저장한 뒤(1038줄), 곧바로 `top_k_val = None`으로 초기화하고 동일한 파싱 코드를 재실행.
→ 1042~1068번 블록 삭제.

### 3. return 이후 도달 불가 코드 (ascends_server.py:1412-1413)
`FileResponse` 반환문 바로 뒤에 `import uvicorn`과 `uvicorn.run(...)` 존재 — 절대 실행되지 않음.
→ 두 줄 삭제.

### 4. 중복 `_unique_preserve` 함수 (ascends_server.py:314, 1415)
동일한 함수가 두 번 정의됨.
→ 1415~1422번 줄 삭제.

---

## 높음 (보안/안정성)

### 5. Path traversal 취약점 (ascends_server.py:925)
```python
# 기존: 문자열 비교 → symlink 등으로 우회 가능
if not str(file_path).startswith(str(pred_dir)):

# 수정: pathlib으로 안전하게
try:
    file_path.relative_to(pred_dir)
except ValueError:
    return HTMLResponse(status_code=404, content="Not found")
```

### 6. 파일 업로드 크기 제한 없음 (ascends_server.py:91 _save_csv)
파일을 통째로 읽기 전에 크기 검증 없음 → 디스크 고갈 가능.
```python
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
content = await file.read()
if len(content) > MAX_UPLOAD_BYTES:
    raise ValueError("File too large. Maximum allowed is 50 MB.")
```

### 7. TOCTOU 경쟁 조건 (ascends_server.py:1591 _unique_run_name)
`exists()` 체크 → `mkdir()` 사이에 타이밍 문제 발생 가능.
```python
# 수정: atomic mkdir으로 교체
while True:
    try:
        (RUNS_DIR / candidate).mkdir(parents=True, exist_ok=False)
        return candidate
    except FileExistsError:
        candidate = f"{base}_{n}"; n += 1
```
주의: 호출부(line 650)의 `out_dir.mkdir(...)` 중복 호출도 함께 제거.

### 8. LAST_TRAIN 무한 증가 (ascends_server.py:1581)
학습 세션마다 dict에 쌓여 메모리 누수 발생.
```python
_LAST_TRAIN_MAX = 20
if ws_id not in LAST_TRAIN and len(LAST_TRAIN) >= _LAST_TRAIN_MAX:
    del LAST_TRAIN[next(iter(LAST_TRAIN))]
```

---

## 중간 (기술 부채)

### 9. 상관관계 계산 코드 중복
`ascends_server.py:108-184`에 직접 구현된 상관관계 로직이 `ascends/core/correlation.py`와 중복.
→ 서버 측 구현 제거하고 core 모듈 import로 통일.

### 10. task 표현 불일치
- CLI/서버: `"r"` / `"c"`
- core 모듈: `"regression"` / `"classification"`
→ 진입점(서버, CLI)에서 `canonicalize_task()` 호출로 통일.

### 11. 예외 처리 너무 광범위 (20곳 이상)
```python
except Exception:
    pass  # 어떤 에러인지 알 수 없음
```
→ 구체적인 예외 타입 + `logger.warning(...)` 추가.

### 12. 거대한 함수들 (각 ~150줄)
- `correlation_run()` — 검증 / 계산 / 저장 / 렌더링 혼재
- `train_run()` — 학습 / 플로팅 / 상태관리 혼재
- `predict_run()` — 검증 / 예측 / I/O 혼재
→ 각 함수를 `_validate_*()`, `_compute_*()`, `_save_*()` 등으로 분리.

### 13. dcor 서브샘플링 시 사용자 알림 없음 (ascends_server.py:165)
5000행 초과 시 조용히 서브샘플링 — 결과가 근사치임을 UI에 표시해야 함.

---

## 낮음 (개선사항)

### 14. `tune_trials` 파라미터 미구현 (ascends/core/train.py:117)
CLI에서 넘기지만 함수 내부에서 사용되지 않음.
→ 하이퍼파라미터 튜닝 구현 또는 파라미터 제거.

### 15. feature alignment 무음 실패 (ascends/core/data.py:68-74)
학습 시 없던 feature가 있으면 경고 없이 0으로 채움.
→ 누락/추가된 feature 목록을 경고 로그로 출력.

### 16. 반환 타입 불일치
`train_eval()`, `train_model()`, `LAST_TRAIN` 캐시가 서로 다른 dict 구조 사용.
→ TypedDict 또는 dataclass로 통일.

### 17. 하드코딩된 값들
| 위치 | 값 | 제안 |
|------|----|------|
| server:68 | `PREVIEW_NROWS = 5` | 설정 파일로 이동 |
| explain.py | `max_samples=500` | 파라미터로 노출 (이미 일부 적용됨) |
| server | `n_estimators=300` | 상수로 분리 |
| server | `dpi=300` / `dpi=220` | 설정 파일로 이동 |

---

## SHAP 관련 (1a2a01c에서 추가된 기능)

### 현황
- `save_default_shap_plot()` 추가 — SHAP beeswarm 플롯 지원 ✅
- SHAP 뷰 전환 (`ascends` / `default`) 추가 ✅
- `/train/shap/view` 엔드포인트 추가 ✅
- non-tree 모델 실패 시 `try/except`로 graceful fallback 처리 ✅

### 잠재적 개선점
- `save_default_shap_plot()`에서 분류 모델의 경우 `shap_values`가 클래스별 리스트로 반환되는데, 현재 `summary_plot`에 그대로 넘기면 다중 클래스에서 경고 또는 의도치 않은 시각화 발생 가능.
- SHAP 계산 중 진행 상황 표시 없음 (대용량 데이터에서 오래 걸림).

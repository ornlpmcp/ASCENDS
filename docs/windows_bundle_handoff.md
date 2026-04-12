# Windows Portable Bundle — Build & Architecture Handoff

이 문서는 ASCENDS Windows 포터블 번들의 구조, 문제 원인, 해결 방법을 설명합니다.
Mac에서 작업하는 기여자가 Windows 번들을 빌드하거나 디버깅할 때 참고하세요.

---

## 배경: Linux/macOS 번들과 무엇이 다른가

Linux/macOS 번들(`make_bundle.sh`)은 `uv` 바이너리를 동봉하고,
런처 스크립트에서 `uv run ascends gui`를 호출합니다.

이 방식이 Linux/macOS에서 잘 작동하는 이유:
- `uv run`은 venv가 손상되어 있어도 자동 복구합니다.
- venv 내부 `python3` 심볼릭 링크가 망가져 있어도, `uv`가 직접 Python을 찾아 재설치합니다.

**Windows에서는 이 방식이 실패합니다.**

---

## 핵심 문제: `pyvenv.cfg`의 절대 경로

Windows에서 uv로 venv를 만들면 `.venv\pyvenv.cfg`에 다음이 기록됩니다:

```
home = C:\Users\developer\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none
```

Windows의 `.venv\Scripts\python.exe`는 실제로 **stub(작은 런처 exe)**입니다.
이 stub은 시작 시 `pyvenv.cfg`를 읽어서 `home` 경로에서 `python313.dll`을 로드합니다.

즉:
1. 번들을 다른 PC에 압축 해제하면 `home` 경로가 존재하지 않습니다.
2. `python313.dll` 로드 실패 → Python 자체가 실행되지 않습니다.
3. `uv run`이 venv를 재구성하려 해도, Python을 인터넷에서 다운로드해야 합니다.

Linux/macOS에서는 `python3`이 실제 실행 파일(또는 실제 파일로 이어지는 심볼릭 링크)이기 때문에 이 문제가 없습니다.

---

## 해결책: Python 배포판을 번들에 통째로 포함

`.venv\Scripts\python.exe` (stub)를 사용하지 않고,
**uv가 관리하는 Python 배포판 전체**(`python.exe`, `python313.dll`, `Lib\`, `DLLs\`)를
번들의 `python\` 폴더에 복사합니다.

런처는 이 `python\python.exe`를 직접 실행하고,
`PYTHONPATH` 환경 변수로 `.venv\Lib\site-packages\`를 지정해 패키지를 찾게 합니다.

이렇게 하면:
- `pyvenv.cfg` 절대 경로 문제가 완전히 우회됩니다.
- 인터넷 연결 없이도 즉시 실행됩니다.
- `uv run`에 의존하지 않습니다.

---

## 번들 디렉토리 구조

```
ASCENDS-v0.4.x-YYYYMMDD-windows/
│
├── python/                        ← uv 관리 Python 배포판 전체 복사
│   ├── python.exe                 ← 진짜 Python 실행 파일
│   ├── python313.dll              ← Python DLL (같은 폴더에 있어야 로드됨)
│   ├── Lib/                       ← Python 표준 라이브러리
│   ├── DLLs/                      ← 확장 모듈 (.pyd 파일들)
│   └── ...
│
├── ASCENDS/
│   ├── .venv/
│   │   └── Lib/site-packages/     ← 모든 패키지 (numpy, pandas, fastapi 등)
│   ├── ascends/                   ← 소스 코드
│   ├── templates/
│   ├── static/
│   ├── examples/
│   ├── ascends_server.py
│   ├── pyproject.toml
│   └── uv.lock
│
├── uv.exe                         ← 동봉 (파워 유저용, 런처에는 불필요)
├── launch_gui.bat                 ← 주 런처 (더블클릭)
├── launch_gui.ps1                 ← PowerShell 런처 (선택)
├── launch_cli.bat                 ← CLI 런처
├── bundle-meta.txt
└── README-BUNDLE.txt
```

---

## 런처 동작 원리

`launch_gui.bat`:

```bat
@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%ASCENDS\.venv\Lib\site-packages"
cd /d "%ROOT%ASCENDS"

echo [ASCENDS] Launching GUI at http://127.0.0.1:7777
echo [ASCENDS] Open your browser at: http://127.0.0.1:7777
echo.

"%ROOT%python\python.exe" -c "import sys; sys.argv[0]='ascends'; from ascends.cli import app; app()" gui %*
```

핵심:
- `PYTHONPATH`로 패키지 경로를 직접 지정 (venv 활성화 불필요)
- `-c "..."` 인라인으로 Typer 진입점 호출 (`pyproject.toml`의 `ascends = "ascends.cli:app"`)
- `sys.argv[0]='ascends'`로 설정해 Typer가 올바른 커맨드명을 인식하게 함
- `gui %*` — `gui` 서브커맨드 + 추가 인자 전달

---

## 빌드 방법

### Windows에서 빌드 (기본)

```powershell
# PowerShell
.\bundle\make_bundle.ps1

# 또는 cmd.exe
bundle\make_bundle.bat
```

### Mac에서 Windows 번들 빌드

Mac에서는 Windows 번들을 **직접 만들 수 없습니다**.
이유: uv가 관리하는 Python 배포판이 플랫폼별로 따로 다운로드되기 때문입니다
(`cpython-3.13-windows-x86_64-none` vs `cpython-3.13-macos-aarch64-none`).

**권장 방법:**

1. **Windows VM 또는 실제 Windows PC에서 빌드**: 가장 확실한 방법

2. **GitHub Actions CI (추천)**: `.github/workflows/`에 Windows runner 잡을 추가해
   PR 머지 또는 태그 생성 시 자동으로 Windows 번들을 빌드하고 릴리스에 첨부

   ```yaml
   jobs:
     build-windows:
       runs-on: windows-latest
       steps:
         - uses: actions/checkout@v4
         - uses: astral-sh/setup-uv@v4
         - run: .\bundle\make_bundle.ps1
           shell: pwsh
         - uses: actions/upload-artifact@v4
           with:
             name: windows-bundle
             path: dist/*.zip
   ```

---

## 빌드 전제 조건

빌드하는 Windows 머신에 필요한 것:

| 항목 | 설명 |
|------|------|
| `uv` | PATH에 있어야 함. [설치 방법](https://docs.astral.sh/uv/getting-started/installation/) |
| 인터넷 연결 | 첫 빌드 시 uv가 Python과 패키지를 다운로드 |
| PowerShell 5+ | Windows 10/11 기본 포함 |

---

## 번들 크기 참고

| 구성 요소 | 압축 전 (대략) |
|-----------|--------------|
| Python 배포판 (`python/`) | ~60 MB |
| `.venv/Lib/site-packages/` | ~400 MB (numpy, torch 제외) |
| 소스 코드 | ~5 MB |
| **zip 압축 후** | **~150-200 MB** |

---

## 트러블슈팅

### `python.exe` 실행 시 "DLL not found" 오류

`python\python313.dll`이 없는 경우입니다.
`make_bundle.ps1`이 `home` 경로를 올바르게 읽었는지 확인:

```powershell
Get-Content .\ASCENDS\.venv\pyvenv.cfg
# home = C:\Users\...\uv\python\cpython-3.13-windows-x86_64-none
# 이 경로가 실제 존재하고 python.exe + python313.dll을 포함하는지 확인
```

### `uv sync` 단계에서 "No Python found"

uv가 해당 Python 버전을 아직 다운로드하지 않은 경우:

```powershell
uv python install 3.13
```

### 번들 실행 시 `ModuleNotFoundError`

`PYTHONPATH`가 제대로 설정됐는지 확인:

```bat
set PYTHONPATH=C:\path\to\bundle\ASCENDS\.venv\Lib\site-packages
python\python.exe -c "import ascends; print('OK')"
```

---

## Linux/macOS 번들과의 차이점 요약

| | Linux/macOS | Windows |
|---|---|---|
| 런처 | `uv run ascends gui` | `python\python.exe` 직접 실행 |
| Python 번들 방식 | 불필요 (uv가 복구) | `python/` 디렉토리 통째로 포함 |
| 패키지 접근 | venv 활성화 | `PYTHONPATH` 직접 지정 |
| 아카이브 형식 | `.tar.gz` | `.zip` |
| 오프라인 실행 | venv 손상 시 재시도 | 완전 오프라인 가능 |

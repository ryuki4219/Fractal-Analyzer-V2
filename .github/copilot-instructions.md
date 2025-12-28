# Fractal-Analyzer-V2: Copilot Instructions

Purpose: Equip AI coding agents to be productive fast in this repo. Focus on project-specific architecture, workflows, and conventions.

## Big Picture
- Streamlit app (`fractal_app.py`) orchestrates data ingest, FD computation, AI prediction, validation, and reporting.
- FD algorithms: optimized vectorized implementations live in `fractal_app.py` (GPU via CuPy if available) and reference versions in `fd_boxcount.py` (`fd_std_boxcount`, `fd_3d_dbc`).
- Learning/prediction: LightGBM (`LGBMRegressor`) maps low-quality image features to high-quality FD. Model + metrics persist across runs.
- Modules:
  - `skin_analysis.py`: face detection (MediaPipe/OpenCV/dlib fallback), region extraction, trouble detection helpers.
  - `experiment_analysis.py`: CSV logging (`experimental_data.csv`), correlation analysis, plotting helpers.
  - Reports: `generate_updated_report.py` (PDF via ReportLab), `scripts/make_template_docx.py` (Word template).

## Workflows
- Run app (Windows, Streamlit): `streamlit run fractal_app.py`.
- Quick env: create venv → `pip install -r requirements.txt` (optional CuPy for GPU).
- CLI FD check: `python fd_boxcount.py <image_path>` to sanity-check FD outputs.
- Paper assets: `python scripts\make_template_docx.py` then edit `docs/templates/Paper_Template.docx`.
- Experiment logging: append rows via `ExperimentDataManager.save_data()` to `experimental_data.csv`.

## Data & Persistence
- Image pairs: naming convention `IMG_XXXX.jpg` ↔ `IMG_XXXX_low1.jpg` (and Low4–7 group). Many UI flows assume this.
- Model: `trained_fd_model.pkl`; History: `training_history.json`. Both auto-loaded/saved by `fractal_app.py`.
- Dataset folder: `SKIN_DATA/` (contains `Facial Skin Condition Dataset.csv` and example subfolders). Paths may include Japanese chars; use robust loaders (OpenCV via buffer read).

## Patterns & Conventions
- Images are BGR (`cv2.imdecode`) internally. Convert with `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)` for FD.
- Data augmentation uses named methods (e.g., `flip_h`, `rotate_180`, `clahe`, `unsharp`, `scale_up/down`) applied symmetrically to high/low pairs.
- FD computation uses multi-scale box sizes; results clipped to 2.0–3.0 with outlier guards.
- GPU toggle is automatic (`cupy` presence → `USE_CUPY=True`); write code that works with `xp` alias (numpy/cupy) and converts via `to_xp`/`to_host`.
- Face regions: prefer MediaPipe landmarks; fallback to OpenCV Haar/dlib; last-resort heuristic rectangle. Keep APIs compatible with `extract_face_regions(image, landmarks)`.

## Integration Points
- Optional modules: `skin_quality_evaluator.py`, `image_quality_assessor.py` are imported conditionally; design new modules to degrade gracefully when missing.
- Streamlit UI: follow existing section patterns—`st.subheader`, `st.metric`, `st.dataframe`, `st.plotly_chart`. Persist session via `st.session_state`.
- Plots: use Matplotlib for static figures; Plotly when `PLOTLY_AVAILABLE`.

## Examples
- Compute FD (2D std-boxcount):
  ```python
  import cv2
  from fd_boxcount import fd_std_boxcount
  img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
  fd, scales, Nh, *_ = fd_std_boxcount(img)
  ```
- Train LightGBM (inside app): build features from low-quality FD/features → `LGBMRegressor(..., n_jobs=-1)`; persist via `pickle.dump(model, open('trained_fd_model.pkl','wb'))`.

## Gotchas
- Use `opencv-python-headless`—no GUI windows; rely on Streamlit/plots for visualization.
- Japanese paths/fonts: prefer buffer-based image loading; ReportLab doc uses Windows fonts (Meiryo/YuGothic/MSMincho) when present.
- Correlation metrics can be `nan` when variance ~0; guard and display user guidance.
- Keep FD slope range sensible; clip/validate to avoid spurious values.

## Quick Commands
- Setup (PowerShell):
  ```powershell
  python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
  ```
- Run app:
  ```powershell
  streamlit run fractal_app.py --server.port 8501
  ```
- Generate Word template:
  ```powershell
  python scripts\make_template_docx.py
  ```

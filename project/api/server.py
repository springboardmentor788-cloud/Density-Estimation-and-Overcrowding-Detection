from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from dataclasses import asdict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import CONFIG
from api.pipeline import discover_models, infer_image_bytes, infer_video_file


PROJECT_ROOT = CONFIG.project_root
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "api"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DeepVision Crowd Monitor API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=str(OUTPUT_ROOT)), name="files")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def models() -> list[dict[str, object]]:
    return [asdict(choice) for choice in discover_models()]


@app.post("/api/infer/image")
async def infer_image(
    file: UploadFile = File(...),
    checkpoint_path: str = Form(...),
    threshold: float = Form(130.0),
    resize_width: int = Form(1280),
    calibration_path: str | None = Form(None),
    prefer_cuda: bool = Form(True),
):
    image_bytes = await file.read()
    normalized_calibration = calibration_path.strip() if calibration_path else None
    if normalized_calibration == "":
        normalized_calibration = None
    try:
        return infer_image_bytes(
            image_bytes,
            checkpoint_path=checkpoint_path,
            calibration_path=normalized_calibration,
            threshold=threshold,
            resize_width=resize_width,
            prefer_cuda=prefer_cuda,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/infer/video")
async def infer_video(
    file: UploadFile = File(...),
    checkpoint_path: str = Form(...),
    threshold: float = Form(130.0),
    resize_width: int = Form(1280),
    sample_fps: float = Form(8.0),
    calibration_path: str | None = Form(None),
    prefer_cuda: bool = Form(True),
):
    session_id = uuid.uuid4().hex
    session_dir = OUTPUT_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
    normalized_calibration = calibration_path.strip() if calibration_path else None
    if normalized_calibration == "":
        normalized_calibration = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_input = Path(tmp.name)

    try:
        summary = infer_video_file(
            temp_input,
            checkpoint_path=checkpoint_path,
            calibration_path=normalized_calibration,
            threshold=threshold,
            resize_width=resize_width,
            sample_fps=sample_fps,
            output_dir=session_dir,
            prefer_cuda=prefer_cuda,
        )
        return {
            "session_id": session_id,
            "frames": summary["frames"],
            "count_mean": summary["count_mean"],
            "count_max": summary["count_max"],
            "count_min": summary["count_min"],
            "overlay_url": f"/files/{session_id}/{Path(summary['overlay_path']).name}",
            "density_url": f"/files/{session_id}/{Path(summary['density_path']).name}",
        }
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)

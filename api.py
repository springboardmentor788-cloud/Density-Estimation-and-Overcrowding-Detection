from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, shutil, os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
  allow_methods=["*"], allow_headers=["*"])

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
  # save file and call your predict.py here
  return {"count": 0, "status": "done"}

@app.get("/results")
async def results(): return []

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)
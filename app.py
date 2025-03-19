from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import uvicorn
from src.models import GCSRequest, ResponseModel
from src.services.mono_service import MonoService


app = FastAPI(title="MONO API", description="API for video processing and analysis")
mono_service = MonoService()

@app.post("/predictions/mono", response_model=ResponseModel)
async def process_video(data: GCSRequest):
    """
    Process video from Google Cloud Storage
    """
    return await mono_service.process_video(data)

@app.post("/upload", response_model=ResponseModel)
async def upload_video(file: UploadFile = File(...)):
    """
    Process uploaded video file
    """
    return await mono_service.process_video(file)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
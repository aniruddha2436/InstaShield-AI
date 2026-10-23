from contextlib import asynccontextmanager
import logging
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from .schema import Profile
from .service import Detector

@asynccontextmanager
async def lifespan(app):
    app.state.detector = Detector()
    yield

app = FastAPI(title='InstaShield AI', version='1.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173','http://127.0.0.1:5173'], allow_methods=['GET','POST'], allow_headers=['Content-Type'])

@app.get('/api/health')
def health(request: Request):
    return {'status':'ready', 'model':'XGBoost', 'threshold':request.app.state.detector.threshold}

@app.post('/api/analyze')
def analyze(profile: Profile, request: Request):
    try:
        return request.app.state.detector.predict(profile)
    except Exception:
        logging.exception('Inference failed')
        raise HTTPException(500, 'Analysis failed. Check the backend console and model dependencies.')

# Production: serve the compiled React app on the same origin as the API.
frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

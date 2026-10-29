import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.database.session import init_db
from app.api.routes import router as api_router

# Ensure static directories exist before mounting
settings.UPLOADS_PATH.mkdir(parents=True, exist_ok=True)
settings.SAMPLES_PATH.mkdir(parents=True, exist_ok=True)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print(f"[FastAPI Lifespan] SQLite database initialized at {settings.DATABASE_URL}")
    print(f"[FastAPI Lifespan] Server ready for requests.")
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        f"College Artificial Neural Networks (ANN) Project #{settings.PROJECT_NUMBER} by {settings.STUDENT_NAME} "
        f"(Reg No: {settings.REGISTRATION_NUMBER}, Roll: {settings.STUDENT_ROLL}) for {settings.SYLLABUS_UNIT} / {settings.COURSE_OUTCOME}. "
        f"Multi-class Food Image Classification using CNN and Mini-Batch Gradient Descent."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files for Uploads and Sample Food Images
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOADS_PATH)), name="uploads")
app.mount("/samples", StaticFiles(directory=str(settings.SAMPLES_PATH)), name="samples")

# Register API Router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Project info metadata endpoint
@app.get("/api/info")
def project_info():
    return {
        "project": settings.PROJECT_NAME,
        "student": settings.STUDENT_NAME,
        "registration_number": settings.REGISTRATION_NUMBER,
        "roll_number": settings.STUDENT_ROLL,
        "unit": settings.SYLLABUS_UNIT,
        "course_outcome": settings.COURSE_OUTCOME,
        "status": "Online",
        "docs": "/docs",
        "api_v1": settings.API_V1_STR
    }

# Mount Frontend Build (SPA support)
from fastapi.responses import FileResponse

frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="frontend_assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Prevent intercepting docs, uploads, samples
        if full_path.startswith(("api", "docs", "redoc", "openapi.json", "uploads", "samples")):
            return {"error": "Not Found"}
        target_file = frontend_dist / full_path
        if full_path and target_file.is_file():
            return FileResponse(str(target_file))
        return FileResponse(str(frontend_dist / "index.html"))
else:
    @app.get("/")
    def root_fallback():
        return project_info()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)


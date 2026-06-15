import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.database import engine, Base
from src.api.inference import ModelInference
from src.api.routes import health, predict, model, train

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables on startup
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    
    # Load model checkpoint
    logger.info("Loading model checkpoint on lifespan startup...")
    inference_engine = ModelInference.get_instance()
    try:
        inference_engine.load()
    except Exception as e:
        logger.error("Failed to load model checkpoint on startup: %s. Inference requests will fail until model loaded.", e)
        
    yield
    
    logger.info("Shutting down API...")

app = FastAPI(
    title="MolPredict Multi-Task GRL API",
    description="Backend API for predicting Tox21 toxicity endpoints and monitoring model training.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development ease
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(predict.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(model.router, prefix="/api/model", tags=["Model Registry"])
app.include_router(train.router, prefix="/api/train", tags=["Training Monitor"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

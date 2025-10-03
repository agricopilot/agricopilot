import os
import logging
import io
from fastapi import FastAPI, Request, Header, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
from vector import query_vector

# ==============================
# Logging
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgriCopilot")

# ==============================
# App Init
# ==============================
app = FastAPI(title="AgriCopilot")

@app.get("/")
async def root():
    return {"status": "AgriCopilot AI Backend is working perfectly"}

# ==============================
# Auth
# ==============================
PROJECT_API_KEY = os.getenv("PROJECT_API_KEY", "agricopilot404")

def check_auth(authorization: str | None):
    if not PROJECT_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    if token != PROJECT_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")

# ==============================
# Global Exception Handler
# ==============================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
    )

# ==============================
# Request Models
# ==============================
class ChatRequest(BaseModel):
    query: str

class DisasterRequest(BaseModel):
    report: str

class MarketRequest(BaseModel):
    product: str

class VectorRequest(BaseModel):
    query: str

# ==============================
# HuggingFace Pipelines
# ==============================
def load_pipeline(task, model_meta, model_fallback):
    """Try Meta model, fallback to public."""
    try:
        return pipeline(task, model=model_meta)
    except Exception as e:
        logger.warning(f"Meta model {model_meta} unavailable. Using fallback {model_fallback}.")
        return pipeline(task, model=model_fallback)

chat_pipe = load_pipeline("conversational",
                          model_meta="meta-llama/Llama-3.1-8B-Instruct",
                          model_fallback="microsoft/DialoGPT-medium")

disaster_pipe = load_pipeline("conversational",
                              model_meta="meta-llama/Llama-3.1-8B-Instruct",
                              model_fallback="microsoft/DialoGPT-medium")

market_pipe = load_pipeline("conversational",
                            model_meta="meta-llama/Llama-3.1-8B-Instruct",
                            model_fallback="microsoft/DialoGPT-medium")

crop_pipe = load_pipeline("image-to-text",
                          model_meta="meta-llama/Llama-3.2-11B-Vision-Instruct",
                          model_fallback="Salesforce/blip-image-captioning-large")

# ==============================
# Helper Functions
# ==============================
def run_conversational(pipe, prompt: str):
    try:
        output = pipe(prompt)
        if isinstance(output, list) and len(output) > 0:
            return output[0].get("generated_text", str(output))
        return str(output)
    except Exception as e:
        logger.error(f"Conversational pipeline error: {e}")
        return f"⚠️ Unexpected model error: {str(e)}"

def run_crop_doctor(image_bytes: bytes, symptoms: str):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prompt = f"Farmer reports: {symptoms}. Diagnose the crop disease and suggest treatment in simple terms."
        output = crop_pipe(image, prompt=prompt)
        if isinstance(output, list) and len(output) > 0:
            return output[0].get("generated_text", str(output))
        return str(output)
    except Exception as e:
        logger.error(f"Crop Doctor pipeline error: {e}")
        return f"⚠️ Unexpected vision model error: {str(e)}"

# ==============================
# Endpoints
# ==============================
@app.post("/crop-doctor")
async def crop_doctor(symptoms: str = Header(...), image: UploadFile = File(...), authorization: str | None = Header(None)):
    check_auth(authorization)
    image_bytes = await image.read()
    diagnosis = run_crop_doctor(image_bytes, symptoms)
    return {"diagnosis": diagnosis}

@app.post("/multilingual-chat")
async def multilingual_chat(req: ChatRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    reply = run_conversational(chat_pipe, req.query)
    return {"reply": reply}

@app.post("/disaster-summarizer")
async def disaster_summarizer(req: DisasterRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    summary = run_conversational(disaster_pipe, req.report)
    return {"summary": summary}

@app.post("/marketplace")
async def marketplace(req: MarketRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    recommendation = run_conversational(market_pipe, req.product)
    return {"recommendation": recommendation}

@app.post("/vector-search")
async def vector_search(req: VectorRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    try:
        results = query_vector(req.query)
        return {"results": results}
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return {"error": f"Vector search error: {str(e)}"}

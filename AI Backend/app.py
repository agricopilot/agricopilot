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
# Setup Logging
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
# AUTH CONFIG
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
try:
    chat_pipe = pipeline("conversational", model="microsoft/DialoGPT-medium")
    disaster_pipe = pipeline("conversational", model="microsoft/DialoGPT-medium")
    market_pipe = pipeline("conversational", model="microsoft/DialoGPT-medium")
    crop_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    logger.info("All pipelines loaded successfully.")
except Exception as e:
    logger.error(f"Error loading pipelines: {e}")
    raise RuntimeError("Pipelines failed to load. Check model availability.")

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
# ENDPOINTS
# ==============================

# 🌱 Crop Doctor
@app.post("/crop-doctor")
async def crop_doctor(
    symptoms: str = Header(...),
    image: UploadFile = File(...),
    authorization: str | None = Header(None)
):
    check_auth(authorization)
    image_bytes = await image.read()
    diagnosis = run_crop_doctor(image_bytes, symptoms)
    return {"diagnosis": diagnosis}

# 🗣 Multilingual Chat
@app.post("/multilingual-chat")
async def multilingual_chat(req: ChatRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    reply = run_conversational(chat_pipe, req.query)
    return {"reply": reply}

# 🌪 Disaster Summarizer
@app.post("/disaster-summarizer")
async def disaster_summarizer(req: DisasterRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    summary = run_conversational(disaster_pipe, req.report)
    return {"summary": summary}

# 🛒 Marketplace Recommendation
@app.post("/marketplace")
async def marketplace(req: MarketRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    recommendation = run_conversational(market_pipe, req.product)
    return {"recommendation": recommendation}

# 🔍 Vector Search
@app.post("/vector-search")
async def vector_search(req: VectorRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    try:
        results = query_vector(req.query)
        return {"results": results}
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return {"error": f"Vector search error: {str(e)}"}

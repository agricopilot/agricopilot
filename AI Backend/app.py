import os
import logging
from fastapi import FastAPI, Request, Header, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.chat_models import ChatHF
from langchain.schema import HumanMessage, AIMessage
from PIL import Image
import io
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
# HuggingFace Chat Models
# ==============================
chat_model = ChatHF(model_name="meta-llama/Llama-3.1-8B-Instruct", temperature=0.3)
disaster_model = ChatHF(model_name="meta-llama/Llama-3.1-8B-Instruct", temperature=0.3)
market_model = ChatHF(model_name="meta-llama/Llama-3.1-8B-Instruct", temperature=0.3)

# Crop Doctor Vision + Language Model
crop_model = ChatHF(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct", temperature=0.3)

# ==============================
# Helper Functions
# ==============================
def run_chat_model(model, prompt: str):
    try:
        response = model([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logger.error(f"Model error: {e}")
        return f"⚠️ Unexpected model error: {str(e)}"

def run_crop_doctor_model(model, image_bytes: bytes, symptoms: str):
    """Send image + text to vision-language model"""
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prompt = f"Farmer reports: {symptoms}. Diagnose the crop disease and suggest treatment in simple language."
        # ChatHF allows messages with image objects as content
        response = model([HumanMessage(content=prompt, additional_kwargs={"image": image})])
        return response.content
    except Exception as e:
        logger.error(f"Crop Doctor model error: {e}")
        return f"⚠️ Unexpected model error: {str(e)}"

# ==============================
# ENDPOINTS
# ==============================
@app.post("/crop-doctor")
async def crop_doctor(symptoms: str = Header(...), image: UploadFile = File(...), authorization: str | None = Header(None)):
    """
    Receives crop image and symptom description.
    Returns diagnosis and suggested treatment.
    """
    check_auth(authorization)
    image_bytes = await image.read()
    result = run_crop_doctor_model(crop_model, image_bytes, symptoms)
    return {"diagnosis": result}

@app.post("/multilingual-chat")
async def multilingual_chat(req: ChatRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    response = run_chat_model(chat_model, req.query)
    return {"reply": response}

@app.post("/disaster-summarizer")
async def disaster_summarizer(req: DisasterRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    response = run_chat_model(disaster_model, req.report)
    return {"summary": response}

@app.post("/marketplace")
async def marketplace(req: MarketRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    response = run_chat_model(market_model, req.product)
    return {"recommendation": response}

@app.post("/vector-search")
async def vector_search(req: VectorRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    try:
        results = query_vector(req.query)
        return {"results": results}
    except Exception as e:
        return {"error": f"Vector search error: {str(e)}"}

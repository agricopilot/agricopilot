# ============================================
# AgriCopilot AI Backend — Optimized Stable Release
# ============================================

import os
import logging
import io
import torch
from fastapi import FastAPI, Request, Header, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
from vector import query_vector

# ==============================
# Logging Setup
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgriCopilot")

# ==============================
# FastAPI App Init
# ==============================
app = FastAPI(title="AgriCopilot")

@app.get("/")
async def root():
    return {"status": "✅ AgriCopilot AI Backend is running and stable."}

# ==============================
# Auth Config
# ==============================
PROJECT_API_KEY = os.getenv("PROJECT_API_KEY", "agricopilot404")

def check_auth(authorization: str | None):
    """Verifies Bearer token for all requests."""
    if not PROJECT_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    if token != PROJECT_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")

# ==============================
# Exception Handler
# ==============================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"error": str(exc)})

# ==============================
# Request Schemas
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
# Hugging Face Config
# ==============================
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    logger.warning("⚠️ No Hugging Face token found. Gated models may fail to load.")
else:
    logger.info("✅ Hugging Face token detected.")

# Device setup (GPU if available)
device = 0 if torch.cuda.is_available() else -1
logger.info(f"🧠 Using device: {'GPU' if device == 0 else 'CPU'}")

# ==============================
# Pipelines
# ==============================
# Conversational + reasoning models (Meta LLaMA)
chat_pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN,
    device=device,
)

disaster_pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN,
    device=device,
)

market_pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN,
    device=device,
)

# Lightweight Meta Vision backbone (ConvNeXt-Tiny)
crop_vision = pipeline(
    "image-classification",
    model="facebook/convnext-tiny-224",
    token=HF_TOKEN,
    device=device,
)

# ==============================
# Helper Functions
# ==============================
def run_conversational(pipe, prompt: str):
    """Handles conversational tasks safely."""
    try:
        output = pipe(prompt, max_new_tokens=200, temperature=0.7, do_sample=True)
        if isinstance(output, list) and len(output) > 0:
            return output[0].get("generated_text", str(output))
        return str(output)
    except Exception as e:
        logger.error(f"Conversational pipeline error: {e}")
        return f"⚠️ Model error: {str(e)}"

def run_crop_doctor(image_bytes: bytes, symptoms: str):
    """
    Hybrid Crop Doctor System:
    1. Uses ConvNeXt to classify plant visuals.
    2. Pulls related info from vector dataset.
    3. LLaMA 3.1 generates a short diagnosis and treatment guide.
    """
    try:
        # --- Step 1: Vision Classification ---
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        vision_results = crop_vision(image)
        if not vision_results or "label" not in vision_results[0]:
            raise ValueError("No vision classification result received.")
        top_label = vision_results[0]["label"]

        # --- Step 2: Vector Knowledge Recall ---
        vector_matches = query_vector(symptoms)
        related_knowledge = " ".join(vector_matches[:3]) if isinstance(vector_matches, list) else str(vector_matches)

        # --- Step 3: Reasoning via LLaMA ---
        prompt = (
            f"A farmer uploaded a maize image showing signs of '{top_label}'. "
            f"Reported symptoms: {symptoms}. "
            f"Knowledge base reference: {related_knowledge}. "
            "Generate a structured diagnostic report with:\n"
            "1. Disease Name\n2. Cause\n3. Treatment\n4. Prevention Tips\n"
            "Keep the explanation short and easy for farmers to understand."
        )

        response = chat_pipe(prompt, max_new_tokens=250, temperature=0.6, do_sample=False, truncation=True)

        # Extract text output
        if isinstance(response, list) and len(response) > 0:
            text = response[0].get("generated_text", "").strip()
            return text if text else "⚠️ No response generated. Try again with clearer image or symptoms."
        return "⚠️ Unexpected response format from reasoning model."

    except Exception as e:
        logger.error(f"Crop Doctor error: {e}")
        return f"⚠️ Crop Doctor encountered an issue: {str(e)}"

# ==============================
# Endpoints
# ==============================
@app.post("/crop-doctor")
async def crop_doctor(
    symptoms: str = Header(...),
    image: UploadFile = File(...),
    authorization: str | None = Header(None)
):
    """Diagnose crop disease from image and text."""
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

# ============================================
# END OF FILE
# ============================================

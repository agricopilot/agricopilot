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
    return {"status": "AgriCopilot AI Backend is running and stable."}

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
    return JSONResponse(status_code=500, content={"error": str(exc)})

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
# HuggingFace Pipelines (with token)
# ==============================
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    logger.warning("⚠️ No Hugging Face token found. Gated models may fail to load.")
else:
    logger.info("✅ Hugging Face token detected.")

# Lightweight vision + reasoning setup
chat_pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN)
disaster_pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN)
market_pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN)

# New lightweight vision backbone (Meta ConvNeXt-Tiny)
crop_vision = pipeline(
    "image-classification",
    model="facebook/convnext-tiny-224",
    token=HF_TOKEN
)

# ==============================
# Helper Functions
# ==============================
def run_conversational(pipe, prompt: str):
    """Handle conversational pipelines safely."""
    try:
        output = pipe(prompt, max_new_tokens=200)
        if isinstance(output, list) and len(output) > 0:
            return output[0].get("generated_text", str(output))
        return str(output)
    except Exception as e:
        logger.error(f"Conversational pipeline error: {e}")
        return f"⚠️ Unexpected model error: {str(e)}"

def run_crop_doctor(image_bytes: bytes, symptoms: str):
    """
    Hybrid Crop Doctor System:
    Combines computer vision (ConvNeXt) + text reasoning (LLaMA) + dataset vector recall.
    Returns a detailed, farmer-friendly diagnosis and treatment.
    """
    try:
        # Step 1: Vision Analysis
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        vision_results = crop_vision(image)
        top_label = vision_results[0]["label"]

        # Step 2: Knowledge Recall via Vector Search
        vector_matches = query_vector(symptoms)
        related_knowledge = " ".join(vector_matches[:3]) if isinstance(vector_matches, list) else str(vector_matches)

        # Step 3: LLaMA Reasoning
        prompt = (
            f"A farmer uploaded a maize image that visually shows signs of {top_label}. "
            f"The farmer also reported the following symptoms: {symptoms}. "
            f"From the knowledge base, related observations include: {related_knowledge}. "
            "Generate a short but clear diagnostic report stating the likely disease, "
            "its cause, treatment method, and simple prevention advice."
        )

        response = chat_pipe(prompt, max_new_tokens=300)
        if isinstance(response, list) and len(response) > 0:
            return response[0].get("generated_text", str(response))
        return str(response)

    except Exception as e:
        logger.error(f"Crop Doctor pipeline error: {e}")
        return f"⚠️ Unexpected model error: {str(e)}"

# ==============================
# ENDPOINTS
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

import os
import logging
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub.utils import HfHubHTTPError
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
    """Validate Bearer token against PROJECT_API_KEY"""
    if not PROJECT_API_KEY:  # If key not set, skip validation
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
class CropRequest(BaseModel):
    symptoms: str

class ChatRequest(BaseModel):
    query: str

class DisasterRequest(BaseModel):
    report: str

class MarketRequest(BaseModel):
    product: str

class VectorRequest(BaseModel):
    query: str

# ==============================
# PROMPTS
# ==============================
crop_template = PromptTemplate(
    input_variables=["symptoms"],
    template="You are AgriCopilot, a multilingual AI assistant created to support farmers. Farmer reports: {symptoms}. Diagnose the most likely disease and suggest treatments in simple farmer-friendly language."
)

chat_template = PromptTemplate(
    input_variables=["query"],
    template="You are AgriCopilot, a supportive multilingual AI guide built for farmers. Farmer says: {query}"
)

disaster_template = PromptTemplate(
    input_variables=["report"],
    template="You are AgriCopilot, an AI disaster-response assistant. Summarize in simple steps: {report}"
)

market_template = PromptTemplate(
    input_variables=["product"],
    template="You are AgriCopilot, an AI agricultural marketplace advisor. Farmer wants to sell or buy: {product}. Suggest best options and advice."
)

# ==============================
# HuggingFace Models
# ==============================
def make_llm(repo_id: str):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",  # conversational for HF models
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        max_new_tokens=1024
    )

crop_llm = make_llm("meta-llama/Llama-3.2-11B-Vision-Instruct")
chat_llm = make_llm("meta-llama/Llama-3.1-8B-Instruct")
disaster_llm = make_llm("meta-llama/Llama-3.1-8B-Instruct")
market_llm = make_llm("meta-llama/Llama-3.1-8B-Instruct")

# ==============================
# ENDPOINT HELPERS
# ==============================
def run_conversational_model(model, prompt: str):
    """Send plain text prompt to HuggingFaceEndpoint and capture response"""
    try:
        logger.info(f"Sending prompt to HF model: {prompt}")
        # Pass prompt as a list of messages for conversational models
        result = model.invoke([{"role": "user", "content": prompt}])
        logger.info(f"HF raw response: {result}")
    except HfHubHTTPError as e:
        if "exceeded" in str(e).lower() or "quota" in str(e).lower():
            return {"parsed": None, "raw": "⚠️ HuggingFace daily quota reached. Try again later."}
        return {"parsed": None, "raw": f"⚠️ HuggingFace error: {str(e)}"}
    except Exception as e:
        return {"parsed": None, "raw": f"⚠️ Unexpected model error: {str(e)}"}

    # Parse output
    parsed_text = None
    if isinstance(result, list) and len(result) > 0 and "content" in result[0]:
        parsed_text = result[0]["content"]
    elif isinstance(result, dict) and "generated_text" in result:
        parsed_text = result["generated_text"]
    else:
        parsed_text = str(result)

    return {"parsed": parsed_text, "raw": result}

# ==============================
# ENDPOINTS
# ==============================
@app.post("/crop-doctor")
async def crop_doctor(req: CropRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    prompt = crop_template.format(symptoms=req.symptoms)
    response = run_conversational_model(crop_llm, prompt)
    return {"diagnosis": response}

@app.post("/multilingual-chat")
async def multilingual_chat(req: ChatRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    prompt = chat_template.format(query=req.query)
    response = run_conversational_model(chat_llm, prompt)
    return {"reply": response}

@app.post("/disaster-summarizer")
async def disaster_summarizer(req: DisasterRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    prompt = disaster_template.format(report=req.report)
    response = run_conversational_model(disaster_llm, prompt)
    return {"summary": response}

@app.post("/marketplace")
async def marketplace(req: MarketRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    prompt = market_template.format(product=req.product)
    response = run_conversational_model(market_llm, prompt)
    return {"recommendation": response}

@app.post("/vector-search")
async def vector_search(req: VectorRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    try:
        results = query_vector(req.query)
        return {"results": results}
    except Exception as e:
        return {"error": f"Vector search error: {str(e)}"}

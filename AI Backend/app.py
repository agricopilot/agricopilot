import os
import logging
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub.utils import HfHubHTTPError
from langchain.schema import HumanMessage
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
PROJECT_API_KEY = "agricopilot404"  # 🔑 Fixed bearer token for hackathon

def check_auth(authorization: str | None):
    """Validate Bearer token against PROJECT_API_KEY"""
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
# MODELS PER ENDPOINT (Meta Models, Conversational)
# ==============================

# 1. Crop Doctor
crop_template = PromptTemplate(
    input_variables=["symptoms"],
    template="You are AgriCopilot, a multilingual AI assistant created to support farmers. Farmer reports: {symptoms}. Diagnose the most likely disease and suggest treatments in simple farmer-friendly language."
)
crop_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
    task="conversational",   # ✅ FIXED
    temperature=0.3,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
    max_new_tokens=1024
)

# 2. Multilingual Chat
chat_template = PromptTemplate(
    input_variables=["query"],
    template="You are AgriCopilot, a supportive multilingual AI guide built for farmers. Farmer says: {query}"
)
chat_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",   # ✅ FIXED
    temperature=0.3,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
    max_new_tokens=1024
)

# 3. Disaster Summarizer
disaster_template = PromptTemplate(
    input_variables=["report"],
    template="You are AgriCopilot, an AI disaster-response assistant. Summarize in simple steps: {report}"
)
disaster_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",   # ✅ FIXED
    temperature=0.3,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
    max_new_tokens=1024
)

# 4. Marketplace Recommendation
market_template = PromptTemplate(
    input_variables=["product"],
    template="You are AgriCopilot, an AI agricultural marketplace advisor. Farmer wants to sell or buy: {product}. Suggest best options and advice."
)
market_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",   # ✅ FIXED
    temperature=0.3,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
    max_new_tokens=1024
)

# ==============================
# ENDPOINTS
# ==============================
@app.post("/crop-doctor")
async def crop_doctor(req: CropRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    prompt = crop_template.format(symptoms=req.symptoms)
    response = crop_llm.invoke([HumanMessage(content=prompt)])
    return {"diagnosis": str(response)}

@app.post("/multilingual-chat")
async def multilingual_chat(req: ChatRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    prompt = chat_template.format(query=req.query)
    response = chat_llm.invoke([HumanMessage(content=prompt)])
    return {"reply": str(response)}

@app.post("/disaster-summarizer")
async def disaster_summarizer(req: DisasterRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    prompt = disaster_template.format(report=req.report)
    response = disaster_llm.invoke([HumanMessage(content=prompt)])
    return {"summary": str(response)}

@app.post("/marketplace")
async def marketplace(req: MarketRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    prompt = market_template.format(product=req.product)
    response = market_llm.invoke([HumanMessage(content=prompt)])
    return {"recommendation": str(response)}

@app.post("/vector-search")
async def vector_search(req: VectorRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    results = query_vector(req.query)
    return {"results": results}

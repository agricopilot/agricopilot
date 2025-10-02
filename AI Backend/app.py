import os
import logging
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub.utils import HfHubHTTPError
from langchain.schema import HumanMessage
from vector import query_vector

# ----------------- CONFIG -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgriCopilot")

PROJECT_API_KEY = os.getenv("PROJECT_API_KEY", "super-secret-123")

# FastAPI app
app = FastAPI(title="AgriCopilot")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- AUTH -----------------
def check_auth(authorization: str | None):
    if not PROJECT_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    if token != PROJECT_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")

# ----------------- REQUEST MODELS -----------------
class CropDoctorRequest(BaseModel):
    symptoms: str

class ChatRequest(BaseModel):
    query: str

class DisasterRequest(BaseModel):
    report: str

class MarketplaceRequest(BaseModel):
    product: str

class VectorRequest(BaseModel):
    query: str

# ----------------- PROMPT TEMPLATES -----------------
crop_template = PromptTemplate(
    input_variables=["symptoms"],
    template="You are AgriCopilot, a multilingual AI crop doctor. Farmer reports: {symptoms}. Diagnose the disease and suggest treatments in simple farmer-friendly language."
)

chat_template = PromptTemplate(
    input_variables=["query"],
    template="You are AgriCopilot, a supportive multilingual AI guide built for farmers. Farmer says: {query}"
)

disaster_template = PromptTemplate(
    input_variables=["report"],
    template="You are AgriCopilot, an AI disaster assistant. Summarize the following report for farmers in simple steps: {report}"
)

market_template = PromptTemplate(
    input_variables=["product"],
    template="You are AgriCopilot, an agricultural marketplace recommender. Farmer wants: {product}. Suggest buyers/sellers and short advice."
)

# ----------------- LLM MODELS -----------------
crop_llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct")
chat_llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct")
disaster_llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct")
market_llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct")

# ----------------- ROOT -----------------
@app.get("/")
async def root():
    return {"status": "✅ AgriCopilot AI Backend running"}

# ----------------- ENDPOINTS -----------------
@app.post("/crop-doctor")
async def crop_doctor(req: CropDoctorRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    try:
        prompt = crop_template.format(symptoms=req.symptoms)
        response = crop_llm.invoke([HumanMessage(content=prompt)])
        return {"success": True, "diagnosis": str(response)}
    except HfHubHTTPError as e:
        if "quota" in str(e).lower():
            return {"success": False, "error": "⚠️ Model quota exceeded. Try again later."}
        raise e

@app.post("/multilingual-chat")
async def multilingual_chat(req: ChatRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    try:
        prompt = chat_template.format(query=req.query)
        response = chat_llm.invoke([HumanMessage(content=prompt)])
        return {"success": True, "reply": str(response)}
    except HfHubHTTPError as e:
        if "quota" in str(e).lower():
            return {"success": False, "error": "⚠️ Model quota exceeded. Try again later."}
        raise e

@app.post("/disaster-summarizer")
async def disaster_summarizer(req: DisasterRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    try:
        prompt = disaster_template.format(report=req.report)
        response = disaster_llm.invoke([HumanMessage(content=prompt)])
        return {"success": True, "summary": str(response)}
    except HfHubHTTPError as e:
        if "quota" in str(e).lower():
            return {"success": False, "error": "⚠️ Model quota exceeded. Try again later."}
        raise e

@app.post("/marketplace")
async def marketplace(req: MarketplaceRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    try:
        prompt = market_template.format(product=req.product)
        response = market_llm.invoke([HumanMessage(content=prompt)])
        return {"success": True, "recommendation": str(response)}
    except HfHubHTTPError as e:
        if "quota" in str(e).lower():
            return {"success": False, "error": "⚠️ Model quota exceeded. Try again later."}
        raise e

@app.post("/vector-search")
async def vector_search(req: VectorRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    try:
        results = query_vector(req.query)
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}

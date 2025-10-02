import os
import logging
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub.utils import HfHubHTTPError
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from vector import query_vector

# ==============================
# Setup Logging
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgriCopilot")

# ==============================
# Config
# ==============================
PROJECT_API_KEY = os.getenv("PROJECT_API_KEY", "super-secret-123")  # 🔑 Change in prod

# ==============================
# App Init
# ==============================
app = FastAPI(title="AgriCopilot")

@app.get("/")
async def root():
    return {"status": "AgriCopilot AI Backend is working perfectly"}

# ==============================
# Global Exception Handler
# ==============================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"error": str(exc)})

# ==============================
# Auth Helper
# ==============================
def check_auth(authorization: str | None):
    if not PROJECT_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    if token != PROJECT_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")

# ==============================
# HuggingFace Model Config
# ==============================
default_model = dict(
    temperature=0.3,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
    max_new_tokens=1024
)

# ==============================
# CHAINS
# ==============================

# 1. Crop Doctor
crop_template = PromptTemplate(
    input_variables=["symptoms"],
    template="You are AgriCopilot, a multilingual AI assistant for farmers. "
             "A farmer reports: {symptoms}. Diagnose the likely disease and suggest "
             "clear, farmer-friendly treatments."
)
crop_llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct", **default_model)
crop_chain = crop_template | crop_llm

# 2. Multilingual Chat
chat_template = PromptTemplate(
    input_variables=["query"],
    template="You are AgriCopilot, a supportive multilingual AI guide. "
             "Respond in the SAME language as the user. Farmer says: {query}"
)
chat_llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", **default_model)
chat_chain = chat_template | chat_llm

# 3. Disaster Summarizer
disaster_template = PromptTemplate(
    input_variables=["report"],
    template="You are AgriCopilot, an AI disaster assistant. "
             "Summarize the following report into 3–5 short, actionable steps farmers can follow: {report}"
)
disaster_llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", **default_model)
disaster_chain = disaster_template | disaster_llm

# 4. Marketplace Recommender
market_template = PromptTemplate(
    input_variables=["product"],
    template="You are AgriCopilot, an agricultural marketplace advisor. "
             "Farmer wants to sell or buy: {product}. Suggest options, advice, and safe trade tips."
)
market_llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", **default_model)
market_chain = market_template | market_llm

# ==============================
# ENDPOINTS (with auth)
# ==============================
@app.post("/crop-doctor")
async def crop_doctor(symptoms: str, authorization: str | None = Header(None)):
    check_auth(authorization)
    try:
        response = crop_chain.invoke({"symptoms": symptoms})
        return {"diagnosis": str(response)}
    except HfHubHTTPError as e:
        return {"error": f"HuggingFace error: {str(e)}"}

@app.post("/multilingual-chat")
async def multilingual_chat(query: str, authorization: str | None = Header(None)):
    check_auth(authorization)
    try:
        response = chat_chain.invoke({"query": query})
        return {"reply": str(response)}
    except HfHubHTTPError as e:
        return {"error": f"HuggingFace error: {str(e)}"}

@app.post("/disaster-summarizer")
async def disaster_summarizer(report: str, authorization: str | None = Header(None)):
    check_auth(authorization)
    try:
        response = disaster_chain.invoke({"report": report})
        return {"summary": str(response)}
    except HfHubHTTPError as e:
        return {"error": f"HuggingFace error: {str(e)}"}

@app.post("/marketplace")
async def marketplace(product: str, authorization: str | None = Header(None)):
    check_auth(authorization)
    try:
        response = market_chain.invoke({"product": product})
        return {"recommendation": str(response)}
    except HfHubHTTPError as e:
        return {"error": f"HuggingFace error: {str(e)}"}

@app.post("/vector-search")
async def vector_search(query: str, authorization: str | None = Header(None)):
    check_auth(authorization)
    try:
        results = query_vector(query)
        return {"results": results}
    except Exception as e:
        return {"error": f"Vector search error: {str(e)}"}

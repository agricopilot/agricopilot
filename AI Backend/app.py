import os
import logging
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from transformers import pipeline

# ==============================
# Logging
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgriCopilot")

# ==============================
# Auth
# ==============================
PROJECT_API_KEY = "agricopilot404"

def check_auth(authorization: str | None):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    if token != PROJECT_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid bearer token")

# ==============================
# FastAPI Init
# ==============================
app = FastAPI(title="AgriCopilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ change to frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Models via Transformers Pipelines
# ==============================
logger.info("Loading models... this may take a while.")

crop_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-11B-Vision-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

chat_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

disaster_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

market_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# ==============================
# Prompt Templates
# ==============================
crop_template = PromptTemplate(
    input_variables=["symptoms"],
    template=(
        "You are AgriCopilot, a multilingual AI crop doctor. "
        "Farmer reports: {symptoms}. Diagnose the issue and suggest treatments "
        "in simple farmer-friendly language."
    )
)

chat_template = PromptTemplate(
    input_variables=["query"],
    template=(
        "You are AgriCopilot, a supportive multilingual assistant for farmers. "
        "Respond in the same language. Farmer says: {query}"
    )
)

disaster_template = PromptTemplate(
    input_variables=["report"],
    template=(
        "You are AgriCopilot, an AI disaster assistant. Summarize this report "
        "into 3–5 clear steps farmers can follow. Report: {report}"
    )
)

market_template = PromptTemplate(
    input_variables=["product"],
    template=(
        "You are AgriCopilot, an AI marketplace advisor. Farmer wants to sell or buy: {product}. "
        "Suggest matches, advice, and safe practices."
    )
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

# ==============================
# Routes
# ==============================
@app.get("/")
async def root():
    return {"status": "AgriCopilot AI Backend running with Transformers"}

@app.post("/crop-doctor")
async def crop_doctor(req: CropRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    prompt = crop_template.format(symptoms=req.symptoms)
    response = crop_pipeline(prompt, max_new_tokens=512, do_sample=True)
    return {"diagnosis": response[0]["generated_text"]}

@app.post("/multilingual-chat")
async def multilingual_chat(req: ChatRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    prompt = chat_template.format(query=req.query)
    response = chat_pipeline(prompt, max_new_tokens=512, do_sample=True)
    return {"reply": response[0]["generated_text"]}

@app.post("/disaster-summarizer")
async def disaster_summarizer(req: DisasterRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    prompt = disaster_template.format(report=req.report)
    response = disaster_pipeline(prompt, max_new_tokens=512, do_sample=True)
    return {"summary": response[0]["generated_text"]}

@app.post("/marketplace")
async def marketplace(req: MarketRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    prompt = market_template.format(product=req.product)
    response = market_pipeline(prompt, max_new_tokens=512, do_sample=True)
    return {"recommendation": response[0]["generated_text"]}

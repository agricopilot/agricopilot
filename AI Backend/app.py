import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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
# MODELS PER ENDPOINT (Meta Models)
# ==============================
crop_template = PromptTemplate(
    input_variables=["symptoms"],
    template="You are an agricultural crop doctor. A farmer reports: {symptoms}. Diagnose the most likely disease and suggest treatments in simple farmer-friendly language."
)
crop_llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-2-7b-chat-hf", task="text-generation")

chat_template = PromptTemplate(
    input_variables=["query"],
    template="You are a multilingual AI assistant for farmers. Answer clearly in the same language as the user. Farmer says: {query}"
)
chat_llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-2-13b-chat-hf", task="text-generation")

disaster_template = PromptTemplate(
    input_variables=["report"],
    template="You are an AI disaster assistant. Summarize the following report for farmers in simple steps: {report}"
)
disaster_llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-2-7b-chat-hf", task="text-generation")

market_template = PromptTemplate(
    input_variables=["product"],
    template="You are an agricultural marketplace recommender. Farmer wants to sell or buy: {product}. Suggest possible matches and advice."
)
market_llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-2-13b-chat-hf", task="text-generation")

# ==============================
# ENDPOINTS
# ==============================
@app.post("/crop-doctor")
async def crop_doctor(symptoms: str):
    prompt = crop_template.format(symptoms=symptoms)
    try:
        response = crop_llm.invoke(prompt)  # ✅ FIXED
        return {"diagnosis": response}
    except HfHubHTTPError as e:
        return {"error": f"HuggingFace error: {str(e)}"}

@app.post("/multilingual-chat")
async def multilingual_chat(query: str):
    prompt = chat_template.format(query=query)
    try:
        response = chat_llm.invoke(prompt)  # ✅ FIXED
        return {"reply": response}
    except HfHubHTTPError as e:
        return {"error": f"HuggingFace error: {str(e)}"}

@app.post("/disaster-summarizer")
async def disaster_summarizer(report: str):
    prompt = disaster_template.format(report=report)
    try:
        response = disaster_llm.invoke(prompt)  # ✅ FIXED
        return {"summary": response}
    except HfHubHTTPError as e:
        return {"error": f"HuggingFace error: {str(e)}"}

@app.post("/marketplace")
async def marketplace(product: str):
    prompt = market_template.format(product=product)
    try:
        response = market_llm.invoke(prompt)  # ✅ FIXED
        return {"recommendation": response}
    except HfHubHTTPError as e:
        return {"error": f"HuggingFace error: {str(e)}"}

@app.post("/vector-search")
async def vector_search(query: str):
    try:
        results = query_vector(query)
        return {"results": results}
    except Exception as e:
        return {"error": f"Vector search error: {str(e)}"}

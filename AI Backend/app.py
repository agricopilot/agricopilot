# app.py
import os
from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from vector import query_vector

app = FastAPI(title="AgriCopilot")

# ==============================
# ROOT HEALTH CHECK
# ==============================
@app.get("/")
async def root():
    return {"status": "AgriCopilot AI Backend is working perfectly"}

# ==============================
# MODELS PER ENDPOINT
# ==============================

# 1. Crop Doctor (Image/Text)
crop_template = PromptTemplate(
    input_variables=["symptoms"],
    template="You are an agricultural crop doctor. A farmer reports: {symptoms}. Diagnose the most likely disease and suggest treatments in simple farmer-friendly language."
)
crop_llm = HuggingFaceEndpoint(repo_id="facebook/bart-large", task="text2text-generation")

# 2. Multilingual Chat
chat_template = PromptTemplate(
    input_variables=["query"],
    template="You are a multilingual AI assistant for farmers. Answer clearly in the same language as the user. Farmer says: {query}"
)
chat_llm = HuggingFaceEndpoint(repo_id="google/mt5-base", task="text2text-generation")

# 3. Disaster Summarizer
disaster_template = PromptTemplate(
    input_variables=["report"],
    template="You are an AI disaster assistant. Summarize the following report for farmers in simple steps: {report}"
)
disaster_llm = HuggingFaceEndpoint(repo_id="google/flan-t5-base", task="text2text-generation")

# 4. Marketplace Recommendation
market_template = PromptTemplate(
    input_variables=["product"],
    template="You are an agricultural marketplace recommender. Farmer wants to sell or buy: {product}. Suggest possible matches and advice."
)
market_llm = HuggingFaceEndpoint(repo_id="tiiuae/falcon-7b-instruct", task="text2text-generation")

# ==============================
# ENDPOINTS
# ==============================

@app.post("/crop-doctor")
async def crop_doctor(symptoms: str):
    prompt = crop_template.format(symptoms=symptoms)
    response = crop_llm(prompt)
    return {"diagnosis": response}

@app.post("/multilingual-chat")
async def multilingual_chat(query: str):
    prompt = chat_template.format(query=query)
    response = chat_llm(prompt)
    return {"reply": response}

@app.post("/disaster-summarizer")
async def disaster_summarizer(report: str):
    prompt = disaster_template.format(report=report)
    response = disaster_llm(prompt)
    return {"summary": response}

@app.post("/marketplace")
async def marketplace(product: str):
    prompt = market_template.format(product=product)
    response = market_llm(prompt)
    return {"recommendation": response}

@app.post("/vector-search")
async def vector_search(query: str):
    results = query_vector(query)
    return {"results": results}

# 🌾 AgriCopilot AI Backend

This repository contains the **backend engine** for **AgriCopilot**, an AI-driven agricultural assistant built with **FastAPI**, **Transformers**, and **LangChain**.  
It provides intelligent conversational endpoints, computer vision crop diagnosis, disaster summarization, marketplace recommendations, and vector-based semantic search.

---

## ⚙️ Tech Stack
- **Framework:** FastAPI  
- **AI Models:** Meta LLaMA 3.1–8B (text) & LLaMA 3.2–11B Vision (multimodal)  
- **Libraries:**  
  - `transformers`, `torch`, `accelerate`, `sentencepiece`  
  - `langchain`, `faiss-cpu`, `langchain-huggingface`, `pandas`  
  - `PIL` (image handling)  
- **Database / Vector Search:** FAISS + LangChain  
- **Deployment Ready For:** Docker, Railway, Render, or AWS EC2  

---

## 🧠 Core Functionalities

| Endpoint | Purpose | Model Used |
|-----------|----------|------------|
| `/multilingual-chat` | General chat & multilingual interaction | `meta-llama/Llama-3.1-8B-Instruct` |
| `/disaster-summarizer` | Disaster analysis and summary | `meta-llama/Llama-3.1-8B-Instruct` |
| `/marketplace` | AI-driven market recommendations | `meta-llama/Llama-3.1-8B-Instruct` |
| `/crop-doctor` | Crop disease diagnosis (image + text) | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| `/vector-search` | FAISS vector search for semantic retrieval | Custom VectorStore |

---

## 🧩 Directory Structure

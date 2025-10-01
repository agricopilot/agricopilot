# prepare_data.py
import os
import kagglehub
import pandas as pd
from datasets import load_dataset

os.makedirs("datasets", exist_ok=True)

# -----------------------
# 1. PlantVillage (Kaggle)
# -----------------------
print("Downloading PlantVillage dataset...")
pv_path = kagglehub.dataset_download("dittakavinikhita/plant-disease-prediction-disease-and-healthy")

# Pick the metadata CSV if available
for file in os.listdir(pv_path):
    if file.endswith(".csv"):
        src = os.path.join(pv_path, file)
        dst = "datasets/plant_disease.csv"
        pd.read_csv(src).to_csv(dst, index=False)
        print("✅ Saved PlantVillage ->", dst)

# -----------------------
# 2. AfriQA (Hugging Face)
# -----------------------
print("Downloading AfriQA dataset...")
afriqa = load_dataset("masakhane/afriqa")
afriqa_df = pd.DataFrame(afriqa["train"])

# Merge question + answer into one text column
afriqa_df["text"] = "Q: " + afriqa_df["question"].astype(str) + " A: " + afriqa_df["answer"].astype(str)
afriqa_df[["text"]].to_csv("datasets/afriqa.csv", index=False)
print("✅ Saved AfriQA -> datasets/afriqa.csv")

# -----------------------
# 3. CrisisNLP (Hugging Face)
# -----------------------
print("Downloading CrisisNLP dataset...")
crisis = load_dataset("QCRI/CrisisBench-all-lang")
crisis_df = pd.DataFrame(crisis["train"])

# Pick relevant columns (tweet_text, label, etc.)
if "tweet_text" in crisis_df.columns:
    crisis_df["text"] = crisis_df["tweet_text"].astype(str)
else:
    crisis_df["text"] = crisis_df.astype(str).agg(" ".join, axis=1)

crisis_df[["text"]].to_csv("datasets/crisis.csv", index=False)
print("✅ Saved CrisisNLP -> datasets/crisis.csv")

print("🎉 All datasets prepared in /datasets")

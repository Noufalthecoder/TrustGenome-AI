from fastapi import FastAPI
import numpy as np
from ai_engine.trust_dna_engine import TrustDNAEngine

app = FastAPI()
engine = TrustDNAEngine()

@app.get("/analyze")
def analyze_wallet(tx_count: int, volume: float):
    features = np.array([[tx_count, volume]])
    dna, score = engine.generate_dna(features)
    return {"TrustDNA": dna[0], "RiskScore": float(score[0])}

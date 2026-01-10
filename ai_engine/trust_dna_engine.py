import numpy as np
from sklearn.ensemble import IsolationForest

class TrustDNAEngine:
    def __init__(self):
        self.model = IsolationForest(contamination=0.15)

    def train(self, features):
        self.model.fit(features)

    def generate_dna(self, features):
        scores = self.model.decision_function(features)
        dna = ["RISKY" if s < -0.15 else "TRUSTED" for s in scores]
        return dna, scores

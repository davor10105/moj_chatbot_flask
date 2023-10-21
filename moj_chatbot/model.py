import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util


def normalize_vector(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


class ChatbotModel:
    def __init__(self):
        self.intent_vectors, self.intent_labels = {}, {}
        self.model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.load()

    def train(self, intent_examples):
        """Trains a chatbot based on intent_examples

        Args:
            intent_examples List<Tuple<String, String>>: List of Tuples (intent id, question)

        Returns:
            None
        """
        for system_id in intent_examples:
            self.intent_vectors[system_id] = []
            self.intent_labels[system_id] = []
            for intent_example in intent_examples[system_id]:
                emb = self.model.encode([intent_example[1]])
                self.intent_vectors[system_id].append(emb)
                self.intent_labels[system_id].append(intent_example[0])
            if len(self.intent_vectors[system_id]) > 0:
                self.intent_vectors[system_id] = np.concatenate(
                    self.intent_vectors[system_id]
                )
        self.persist()

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        with open(os.path.join("data/intent_vectors.pickle"), "wb") as f:
            pickle.dump(self.intent_vectors, f)
        with open(os.path.join("data/intent_labels.pickle"), "wb") as f:
            pickle.dump(self.intent_labels, f)

    def load(self):
        """Loads trained component"""
        try:
            with open(os.path.join("data/intent_vectors.pickle"), "rb") as f:
                self.intent_vectors = pickle.load(f)
            with open(os.path.join("data/intent_labels.pickle"), "rb") as f:
                self.intent_labels = pickle.load(f)

        except:
            self.intent_vectors = {}
            self.intent_labels = {}

    def query(self, system_id, question, top_k):
        """Queries the chatbot"""

        key_embs = self.intent_vectors[system_id]
        query_emb = self.model.encode([question])
        scores = util.dot_score(normalize_vector(query_emb), normalize_vector(key_embs))
        scores = (scores[0] + 1) / 2
        sorted_indices = np.argsort(-scores)[:top_k]
        retval = []
        for index in sorted_indices:
            retval.append((self.intent_labels[system_id][index], scores[index].item()))

        return retval

import os
import pickle
import torch
import torch.nn.functional as F
from transformers import pipeline


class ChatbotModel:
    def __init__(self):
        self.intent_vectors, self.intent_labels = [], []
        self.extractor = pipeline(
            # model="data/paraphrase_model",
            model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            task="feature-extraction",
        )
        self.load()

    def train(self, intent_examples):
        """Trains a chatbot based on intent_examples

        Args:
            intent_examples List<Tuple<String, String>>: List of Tuples (intent id, question)

        Returns:
            None
        """
        self.intent_vectors, self.intent_labels = [], []
        for intent_example in intent_examples:
            emb = self.extractor([intent_example[1]], return_tensors=True)[0][:, 0]
            self.intent_vectors.append(emb)
            self.intent_labels.append(intent_example[0])
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

        except ValueError:
            self.intent_vectors = []
            self.intent_vectors = []

    def query(self, question, top_k):
        """Queries the chatbot"""

        key_embs = torch.cat(self.intent_vectors)
        query_emb = self.extractor([question], return_tensors=True)[0][:, 0]
        scores = F.normalize(query_emb) @ F.normalize(key_embs).T
        scores = scores[0]
        sorted_indices = torch.argsort(scores, descending=True)[:top_k]
        retval = []
        for index in sorted_indices:
            retval.append((self.intent_labels[index], scores[index].item()))

        return retval

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from unidecode import unidecode
import copy
import psycopg2
from postgres_secrets import *


def normalize_vector(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


class ChatbotModel:
    def __init__(self):
        self.embedding_db = {}
        self.model = SentenceTransformer(
            "./data/model_data/sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2",  # "paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.load()

    def train(self, intent_examples):
        """Trains a chatbot based on intent_examples

        Args:

        Returns:
            None
        """
        for system_example in intent_examples:
            system_id = system_example["SystemID"]
            edited_system_dict = {}
            if system_id in self.embedding_db:
                edited_system_dict = copy.deepcopy(self.embedding_db[system_id])
            # deletion
            for deleted_item in system_example["DeletedItems"]:
                if deleted_item["QuestionID"] in edited_system_dict:
                    del edited_system_dict[deleted_item["QuestionID"]]
                    del edited_system_dict[deleted_item["QuestionID"] + "###clean"]
            # addition and edit
            for added_item in (
                system_example["AddedItems"] + system_example["EditedItems"]
            ):
                emb = self.model.encode([added_item["QuestionText"]])
                edited_system_dict[added_item["QuestionID"]] = {
                    "IntentID": added_item["IntentID"],
                    "Embedding": emb,
                }
                ##### embed cleaned text
                emb = self.model.encode([unidecode(added_item["QuestionText"])])
                edited_system_dict[added_item["QuestionID"] + "###clean"] = {
                    "IntentID": added_item["IntentID"],
                    "Embedding": emb,
                }
            self.embedding_db[system_id] = edited_system_dict

        self.persist()

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        with open(os.path.join("data/embedding_db.pickle"), "wb") as f:
            pickle.dump(self.embedding_db, f)

    def load(self):
        """Loads trained component"""
        try:
            with open(os.path.join("data/embedding_db.pickle"), "rb") as f:
                self.embedding_db = pickle.load(f)

        except:
            # self.embedding_db = {}
            self.reload()

    def reload(self):
        conn = psycopg2.connect(
            dbname=DBNAME,
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
        )
        cur = conn.cursor()
        cur.execute(f"SELECT default_version, system_id FROM versions")
        default_versions = cur.fetchall()
        for default_version in default_versions:
            version_date, system_id = default_version
            version_date, system_id = str(version_date), str(system_id)
            cur.execute(
                f"SELECT question_id, question, intent_id from questions_{system_id}_{version_date}"
            )
            questions = cur.fetchall()
            formatted_questions = []
            for question in questions:
                formatted_question = {
                    "IntentID": str(question[2]),
                    "QuestionID": str(question[0]),
                    "QuestionText": str(question[1]),
                }
                formatted_questions.append(formatted_question)
            self.train(
                [
                    {
                        "SystemID": system_id,
                        "AddedItems": formatted_questions,
                        "EditedItems": [],
                        "DeletedItems": [],
                    }
                ]
            )
        cur.close()
        conn.close()

    def query(self, system_id, question, top_k):
        """Queries the chatbot"""

        key_embs = np.concatenate(
            [
                questionItem["Embedding"]
                for questionID, questionItem in self.embedding_db[system_id].items()
            ]
        )
        intent_ids = [
            questionItem["IntentID"]
            for questionID, questionItem in self.embedding_db[system_id].items()
        ]
        query_emb = self.model.encode([question])
        scores = util.dot_score(normalize_vector(query_emb), normalize_vector(key_embs))
        scores = (scores[0] + 1) / 2
        sorted_indices = np.argsort(-scores)
        retval = []
        currentIntent = set()
        currentIntentNum = 0
        for index in sorted_indices:
            selectedIntent = intent_ids[index]
            if len(currentIntent) == 0 or selectedIntent not in currentIntent:
                retval.append((selectedIntent, scores[index].item()))
                currentIntent.add(selectedIntent)
                currentIntentNum += 1
            if currentIntentNum == top_k:
                break

        return retval

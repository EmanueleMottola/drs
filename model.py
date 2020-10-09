
############################################
# Creation of the class drsBERT
############################################
import pickle
from sentence_transformers import SentenceTransformer
import torch
import database
import logging
from tqdm import tqdm
import numpy as np
import scipy
logging.basicConfig(level=logging.DEBUG)


class drsBERT:

    __instance = None
    encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    @staticmethod
    def getInstance(self):
        if drsBERT.__instance == None:
            drsBERT()

        return drsBERT.__instance

    def __init__(self):

        if drsBERT.__instance != None:
            raise Exception("Singleton class!!")
        else:
            drsBERT.__instance = self


    # def load_embeddings(self):
    #     # logging.debug("model: load_embeddings: loading embedings..")
    #     # files_s_embeddings = database.retrieve_embeddings_from_pickle()
    #     # sentences = []
    #     # id = 0
    #     # for doc in files_s_embeddings:
    #     #     for emb in doc['embeddings']:
    #     #         sentences.append((id, emb))
    #     #         id += 1
    #     # logging.debug("model: load_embeddings: loaded")
    #     # return sentences
    #
    #     logging.debug("Loading embeddings.")
    #
    #     count = database.count_sentences()
    #     conn = database.connect()
    #     cur = conn.cursor()
    #
    #     sentences_embeddings = []
    #     id_vect = []
    #
    #     for index in tqdm(range(1, count + 1)):
    #         id, vect = database.retrieve_embedding(cur, index)
    #         vect = np.array(vect)
    #         sentences_embeddings.append(vect/np.linalg.norm(vect))
    #         id_vect.append(id)
    #
    #     sentences_embeddings = np.asmatrix(sentences_embeddings)
    #     logging.debug("Shape of matrix is ")
    #     logging.debug(sentences_embeddings.shape)
    #     return sentences_embeddings, id_vect

    def convert_query_to_embedding(self, query):
        return self.encoder.encode(query, show_progress_bar=False)
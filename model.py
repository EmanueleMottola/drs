
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

    def convert_query_to_embedding(self, query):
        return self.encoder.encode(query, show_progress_bar=False)

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
    def __init__(self, name):
        self.name = name
        self.PATH_TO_SENTENCE_TRANSFORMER = 'C:/Users/Mottola/Documents/InfineonSearchEngine/infineonSentenceBERT'
        self.PICKLE_DATA_JSON = 'C:/Users/Mottola/Documents/InfineonSearchEngine/sentence_embeddings.pickle'
        self.PATH_TO_PDFs = 'C:/Users/Mottola/Documents/Thesis_local/text_corpus/pdf'
        # self.files_sentence_embeddings = self.load_embeddings()
        '''[{
            "text": "eccolo",
            "title": "sticazzi",
            "hook": "fantasticissimo",
            "sentences": ["ciao", "bello"],
            "embeddings": [np.arange(768,), np.arange(768,)]
        }]'''
        self.encoder = self.load_Sentence_Transformer()

    def load_embeddings(self):
        # logging.debug("model: load_embeddings: loading embedings..")
        # files_s_embeddings = database.retrieve_embeddings_from_pickle()
        # sentences = []
        # id = 0
        # for doc in files_s_embeddings:
        #     for emb in doc['embeddings']:
        #         sentences.append((id, emb))
        #         id += 1
        # logging.debug("model: load_embeddings: loaded")
        # return sentences

        logging.debug("Loading embeddings.")

        count = database.count_sentences()
        conn = database.connect()
        cur = conn.cursor()

        sentences_embeddings = []
        id_vect = []

        for index in tqdm(range(1, count + 1)):
            id, vect = database.retrieve_embedding(cur, index)
            vect = np.array(vect)
            sentences_embeddings.append(vect/np.linalg.norm(vect))
            id_vect.append(id)

        sentences_embeddings = np.asmatrix(sentences_embeddings)
        logging.debug("Shape of matrix is ")
        logging.debug(sentences_embeddings.shape)
        return sentences_embeddings, id_vect




    def load_Sentence_Transformer(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        model.to(device)
        return model

import numpy as np
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjectionTree, RandomBinaryProjections, RandomDiscretizedProjections
from nearpy.distances import CosineDistance
from nearpy.filters import UniqueFilter, NearestFilter
import model, service
import time
import database
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)

# print("loading model..")
# infineonBERT = model.drsBERT("Infineon search engine")
#
#
# print("loading nearpy..")
# # Dimension of our vector space
# dimension = 768
# # Create a random binary hash with 10 bits
# rbp = RandomBinaryProjections('rbp', 12)
# # Create engine with pipeline configuration
# engine = Engine(dimension, lshashes=[rbp])
# # Index 1000000 random vectors (set their data to a unique string)
#
# # Create random query vector
# query = service.convert_query_to_embedding_Sentence_BERT(infineonBERT.encoder, ["What is the CO2 balance of Infineon Supply Chain?"])
# print(query[0].shape)
#
# print("Indexing vectors...", end="")
# for file in infineonBERT.files_sentence_embeddings:
#     for sent, vect in zip(file['sentences'], file['embeddings']):
#         engine.store_vector(np.reshape(vect, 768), sent)
#
# print(" indexed!")
#
#
#
# # Get nearest neighbours
#
# start = time.time()
#
# N = engine.neighbours(np.reshape(query[0], 768))
#
# finish = time.time() - start
#
# for vect, data, dist in N:
#     print(data)
#
# print(finish)

# def create_LSH_index(files_sentence_embeddings):
#     dimension = 768
#     # Create a random binary hash with 10 bits
#     rbp = RandomBinaryProjections('rbp', 10)
#     # Create engine with pipeline configuration
#     engine = Engine(dimension, lshashes=[rbp], distance=CosineDistance())
#
#
#     # Index sentence vectors (set their data to a unique string)
#     print("Indexing vectors...", end="")
#     for (id, vect) in files_sentence_embeddings:
#         engine.store_vector(np.reshape(np.asarray(vect, dtype=np.float32), 768), id)
#
#     print(" indexed!")
#     return engine

def create_LSH_index_step():
    dimension = 768
    unique = UniqueFilter()
    nearest = NearestFilter(20)
    # Create a random binary hash with 10 bits
    rbp = RandomDiscretizedProjections('default', 10, bin_width=5, rand_seed=4321)
    # Create engine with pipeline configuration
    engine = Engine(dimension, lshashes=[rbp], distance=CosineDistance())


    # Index sentence vectors (set their data to a unique string)

    count = database.count_sentences()
    conn = database.connect()
    cur = conn.cursor()
    logging.debug("Loading embeddings from database.")
    for index in tqdm(range(1, count+1)):
        id, vect = database.retrieve_embedding(cur, index)
        engine.store_vector(np.reshape(np.asarray(vect, dtype=np.float32), 768), id)

    conn.close()
    logging.debug("Embeddings loaded.")
    return engine

def search_with_LSH(engine, list_query_embeds):
    # N = tupla con vettore, id e distanza
    N = engine.neighbours(np.reshape(list_query_embeds[0], 768))
    doc_sent_title = database.retrieve_sentence_and_doctitle(N)
    logging.debug(doc_sent_title)
    render_data_SBERT = []

    for index, (doc, sent, title) in enumerate(doc_sent_title):
        dist = N[index][2] # retrieve the cosine distance
        render_data_SBERT.append({
            "id": doc,
            "title": title,
            "text": sent,
            "score": dist
        })

    return render_data_SBERT
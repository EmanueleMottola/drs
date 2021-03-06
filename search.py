import scipy
import prettytable
import textwrap
import os
import threading
import math
import numpy as np
import logging
import database
import time


logging.basicConfig(level=logging.DEBUG)
NUM_THREADS = 1
NUM_DOCUMENTS = int(50 / NUM_THREADS)


def search(queries, query_embeds, files_converted):

    # list containing for every document the best sentence match
    best_sent_document = []
    #print(queries)
    #print(query_embeds)

    for query, query_embed in zip(queries, query_embeds):
        for document in files_converted:

            distances = scipy.spatial.distance.cdist([query_embed], document['embeddings'], "cosine")[0]
            distances = zip(range(len(distances)), distances)
            distances = sorted(distances, key=lambda x: x[1])
            best_sent_document.append((document['hook'], # cambiamento per gli embeddings con sentence_articles.pickle
                                       ''.join(x for x in document['sentences'][distances[0][0]:distances[0][0] + 5]),
                                       distances[0][1]))
            # at this point, for each document: distances has the sentences ranked by cosine similarity
        documents = sorted(best_sent_document, key=lambda x: x[2])

        results = []
        for count, entry in enumerate(documents[0:30]):
            doc = {}
            doc['id'] = count+1
            doc['title'] = os.path.splitext(entry[0])[0]
            doc['text'] = entry[1]
            doc['score'] = round(1 - entry[2], 4)
            results.append(doc)
    #showResults(results)
    return results

def multiprocess_search(query_embeds, sentences_embeddings, id_vect):

    #logging.debug("Multiprocess search.")
    best_sent_documents = []
    results = {}
    threads = list()
    num_files = math.ceil(len(sentences_embeddings) / NUM_THREADS)
    lock = threading.Lock()
    for query_embed in query_embeds:
        query_embed = np.array(query_embed)
        query_embed = query_embed / np.linalg.norm(query_embed)
        for index in range(NUM_THREADS):
            index_left = index*num_files
            index_right = (index+1) * num_files
            if index_right > len(sentences_embeddings):
                index_right = len(sentences_embeddings)
            x = threading.Thread(target=search_in_doc, args=(query_embed,
                                                             sentences_embeddings[index_left:index_right],
                                                             best_sent_documents,
                                                             id_vect[index_left:index_right],
                                                             lock))
            threads.append(x)
            x.start()

        for thread in threads:
            thread.join()


        best_sent_documents = sorted(best_sent_documents, key=lambda x: x[0], reverse=True)
        #logging.debug(best_sent_documents)
        doc_sent_title = database.retrieve_sentence_and_doctitle(best_sent_documents)
        #logging.debug(doc_sent_title)
        set_doc_id = set()
        for ((doc_id, sentence, title), (similarity, embedding)) in zip(doc_sent_title, best_sent_documents):
            new_entry = {
                "id": doc_id,
                "title": title,
                "text": sentence,
                "score": similarity
            }
            if doc_id in results.keys():
                results[doc_id]["text"] += " " + new_entry["text"]
                results[doc_id]["score"] += new_entry["score"]
            else:
                results[doc_id] = new_entry

        results = sorted(results.values(), key=lambda x: x['score'], reverse=True)

    return list(results)

# NOTE: this function needs the sentence embeddings to be normalized when loaded from DB
def search_in_doc(query_embed, sentence_embeddings, best_doc, id_vect, lock):


    # logging.debug("Thread started.")

    # make the transpose for the dot product
    start = time.time()
    sentence_embeddings = sentence_embeddings.transpose()
    check = time.time() - start
    # logging.debug(check)

    # compute the distances using the dot product
    distances = np.dot(query_embed, sentence_embeddings)
    distances = np.squeeze(np.asarray(distances))
    #
    # for vect in sentence_embeddings:
    #     distances.append(np.dot(query_embed, vect))
    #distances = scipy.spatial.distance.cdist([query_embed], sentence_embeddings, "cosine")[0]
    finish = time.time() - start
    #logging.debug(finish)

    # sort the sentences based on the similarity
    distances = zip(distances, id_vect)
    distances = sorted(distances, key=lambda x: x[0], reverse=True)
    # logging.debug("Thread finished.")

    # store in the data structure common to all the threads
    lock.acquire()
    print(NUM_DOCUMENTS)
    for i in range(50):
        best_doc.append(distances[i])
    lock.release()


def showResults(res):
    table = prettytable.PrettyTable()
    table.field_names = ["Title", "Text", "Score"]
    for el in res:
        title = el[1]
        title = textwrap.fill(title, width=35)
        text = el[2]
        text = textwrap.fill(text, width=75)
        text = text + '\n\n'
        score = el[3]
        table.add_row([title, text, score])

    print(str(table))
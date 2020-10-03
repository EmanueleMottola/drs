from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import threading
import math
import os

NUM_THREADS = 10

def start_client():
    config = {
        'host': '127.0.0.1',
        'port': '9200'
    }
    c = Elasticsearch([config, ], maxsize=25, timeout=30000)
    print("connected")
    return c

def index_documents(cl, documents):
    cl.indices.delete(index='ifx-doc', ignore=[400, 404])
    mapping = {
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768
                },
                "title": {
                    "type": "text"
                },
                "sentence": {
                    "type": "text"
                }
            }
        }
    }
    cl.indices.create(index='ifx-doc', body=mapping)


    id=0
    sents = []
    for doc in documents:
        for sent, emb in zip(doc['sentences'], doc['embeddings']):
            try:
                print(sent)
                print(doc['hook'])
                new_doc = {'embedding': emb, 'sentence': sent, 'title': doc['hook'], "_op_type": "index",
                           "_index": 'ifx-doc'}
                id += 1
                if id % 1000 == 0:
                    print(id)
                sents.append(new_doc)
            except UnicodeEncodeError:
                continue
    print("Indexing doucuments to ElasticSearch..")
    bulk(cl, sents)
    print("done")


def search(cl, query_vector):
    script_query = {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    print("script fatto")
    response = cl.search(
        index='ifx-doc',
        body={
            "size": 50,
            "query": script_query
        }
    )
    documents = []
    keep = set()
    for hit in response["hits"]["hits"]:
        doc = {'id': hit["_id"], 'title': os.path.splitext(hit["_source"]["title"])[0], 'text': hit["_source"]["sentence"], 'score': str(round((hit["_score"]-1)*100, 2))+'%'}
        if doc['title'] in keep:
            for d in documents:
                if d['title'] == doc['title']:
                    d['text'] = d['text'] + "\n" + doc['text']
                    break
        else:
            keep.add(doc['title'])
            documents.append(doc)

    return documents
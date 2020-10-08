# drs

*drs* is a document retrieval system to search for documents related to a query. 
It exploits state-of-the-art NLP techniques to find semantically similar sentences - 
and documents they belong to.

The web application computes the *cosine similarity* between the query and all the sentences 
of the documents, ranking the results accordingly.


# Reference
[1] - *[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)*, Nils Reimers, Iryna Gurevych (EMNLP 2019)
import spacy
import os
import json
import model

PATH_TO_JSON_DOCUMENTS = "/home/emanuele/Documents/cord-19/pdf_json/"


def iterate_over_files():
    for i, entry in enumerate(os.listdir(PATH_TO_JSON_DOCUMENTS)):
        fp = open(PATH_TO_JSON_DOCUMENTS + entry)
        document = json.load(fp)


        print(document)
        if i == 1:
            break


# INPUT: string
#
# OUTPUT: {'sentence': string,
#           'embedding': np.array}
def sentence_tokenization(article):

    paragraph = []

    # tokenization
    doc = nlp(article)
    for sent in doc.sents:
        print(sent)
        paragraph.append({'sentence': sent, 'embedding': model.drsBERT.convert_query_to_embedding(drsBERT, [str(sent)])})

    return paragraph

if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    drsBERT = model.drsBERT()
    iterate_over_files()

    sentence_tokenization("This is a hard time. With Corona, we don't know what to do. What would you do?")




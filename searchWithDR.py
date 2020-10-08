import requests
import pandas as pd
import re
import owlready2
import spacy
import time
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

PATH_TO_ONTOLOGY = "C:\\Users\\Mottola\\Documents\\ontologies\\DigitalReferenceUpdatedDescriptionNoIndividuals.owl"
PATH_SAVED_ONTOLOGY = "C:\\Users\\Mottola\\Documents\\ontologies\\AnnotatedDigitalReference.owl"

# query the DR to retrieve all the triples with ObjectProperty
def queryDigitalReference():
    response = requests.post('http://localhost:3030/DigitalReference/sparql',
                         data = {
                             'query': """prefix owl: <http://www.w3.org/2002/07/owl#> 
                                prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>


                                SELECT DISTINCT ?domain ?dl ?dd ?predicate ?range ?rl ?rd 
                                WHERE {
                                  ?predicate a owl:ObjectProperty .
                                  ?predicate rdfs:domain ?domain .
                                  ?predicate rdfs:range ?range .
                                  ?domain a owl:Class .
                                  ?range a owl:Class .
                                  FILTER (!isBlank(?predicate))
                                  FILTER (!isBlank(?domain))
                                  FILTER (!isBlank(?range))
                                  OPTIONAL { ?domain rdfs:label ?domlabel .
                                    BIND(STR(?domlabel) as ?dl)}
                                  OPTIONAL { ?domain rdfs:comment ?domdescription .
                                    BIND(STR(?domdescription) as ?dd)}
                                  OPTIONAL { ?range rdfs:label ?rangelabel .
                                    BIND(STR(?rangelabel) as ?rl)}
                                  OPTIONAL { ?range rdfs:comment ?rangedescription .
                                    BIND(STR(?rangedescription) as ?rd)}

                                }"""
                         })
    return response.json()


# retrieve the response and organizes it in a data frame
def create_data_frame(response_list):
    domain = [] # domain entity
    dl = [] # domain label
    dd = [] # domain description
    predicate = [] # predicate entity
    range = [] # range entity
    rl = [] # range label
    rd = [] # range description
    for x in response_list['results']['bindings']:
        #try:
        domain.append(x[response_list['head']['vars'][0]]['value'])
        #except KeyError:
        #    domain.append("")
        try:
            dl.append(x[response_list['head']['vars'][1]]['value'])
        except KeyError:
            dl.append("")
        try:
            dd.append(x[response_list['head']['vars'][2]]['value'])
        except KeyError:
            dd.append("")
        #try:
        predicate.append(x[response_list['head']['vars'][3]]['value'])
        #except:
        #    predicate.append("")
        #try:
        range.append(x[response_list['head']['vars'][4]]['value'])
        #except KeyError:
        #    range.append("")
        try:
            rl.append(x[response_list['head']['vars'][5]]['value'])
        except:
            rl.append("")
        try:
            rd.append(x[response_list['head']['vars'][6]]['value'])
        except:
            rd.append("")

    dict = {
        response_list['head']['vars'][0]: domain,
        response_list['head']['vars'][1]: dl,
        response_list['head']['vars'][2]: dd,
        response_list['head']['vars'][3]: predicate,
        response_list['head']['vars'][4]: range,
        response_list['head']['vars'][5]: rl,
        response_list['head']['vars'][6]: rd
    }
    return pd.DataFrame(dict)

# extracts the name of the entities, together with getSplittedText
def getClassName(classEntity):
    if len(classEntity.split("#")) == 2:
        return classEntity.split("#")[1]
    else:
        return "NOT A NAME"


def getSplittedText(name):
    # convert CamelCase and  syntax into normal strings
    splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
    splitted = [x.replace("_", " ") for x in splitted]
    return ' '.join(splitted)


# create the sentences out of triples
def create_string_from_triples(df):
    queries = []
    for index, row in df.iterrows():
        subject = getSplittedText(getClassName(row[0]))
        predicate = getSplittedText(getClassName(row[3]))
        obj = getSplittedText(getClassName(row[4]))
        sentence = subject + " " + predicate + " " + obj
        queries.append((owlready2.IRIS[row[0]], sentence))

    return queries


def importOntology():
    onto = owlready2.get_ontology(PATH_TO_ONTOLOGY).load()
    return onto


def insert_annotations(onto, entity, annotations):
    for doc in annotations:
        entity.seeAlso.append(doc['title'])


def extract_keywords_from_query(query):
    nlp = spacy.load("en_core_web_sm")
    stop_words = nlp.Defaults.stop_words
    stop_words.add('?')
    stop_words.add(',')
    stop_words.add('.')
    stop_words.add('!')
    start = time.time()
    # sentence = "which is the relation between the digital reference and semantic web?"
    # sentence1 = "how can heatmap and convolutional neural networks be related?"
    doc = nlp(query.lower())
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    print("Noun phrases:", noun_phrases)
    remove_stop_words = [[word if str(word) not in stop_words else ',' for word in nlp(sent)] for sent in noun_phrases]
    remove_stop_words = [' '.join(str(x) for x in sent).split(',') for sent in remove_stop_words]
    final_list = []
    for a in remove_stop_words:
        final_list += [x.strip() for x in list(filter(lambda x: x != '', a))]
    print("Final result", final_list)
    print(time.time() - start)
    return final_list

def search_class_annotations(keyword, digital_reference_classes_and_names):
    documents = set()
    for (entity, name) in digital_reference_classes_and_names:
        if fuzz.ratio(keyword, name) > 80:
            print("Fuzzy match: DR " + name + "- Key: " + keyword)
            documents.update(entity.comment)
    print(documents)
    return documents

'''
#if __name__ == "__main__":

    # load queries
    onto = importOntology()
    response_json = queryDigitalReference()
    df = create_data_frame(response_json)
    queries = create_string_from_triples(df)
    print(queries)

    # load model and connect to elasticsearch
    infineonBERT = model.drsBERT("Infineon search engine")
    client = estastic.start_client()

    for (entity, text) in queries:
        list_query_embeds = service.convert_query_to_embedding_Sentence_BERT(infineonBERT.encoder, [text])
        render_data = estastic.search(client, list_query_embeds[0])
        insert_annotations(onto, entity, render_data)

    with open(PATH_SAVED_ONTOLOGY, "wb") as fp:
        onto.save(file=fp)
'''
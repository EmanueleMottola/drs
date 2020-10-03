import service
import owlready2
import time
import re

import spacy

PATH_TO_ONTOLOGY = "C:\\Users\\Mottola\\Documents\\ontologies\\DigitalReferenceUpdatedDescriptionNoIndividuals.owl"
PATH_SAVED_ONTOLOGY = "C:\\Users\\Mottola\\Documents\\ontologies\\AnnotatedDigitalReference.owl"


def importOntology():
    onto = owlready2.get_ontology(PATH_TO_ONTOLOGY).load()
    return onto


def getClasses(ontology):
    return list(ontology.classes())


def getProperties(ontology):
    return [x for x in list(ontology.properties())]


def getClassName(classes):
    # access the uris of the classes, and split them to get the classNames if they exist after the #
    return [(x, x.iri.split("#")[1]) for x in classes if len(x.iri.split("#")) == 2]


def getSplittedText(name):
    # convert CamelCase and  syntax into normal strings
    splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
    splitted = [x.replace("_", " ") for x in splitted]
    return ' '.join(splitted)

def get_tuple_class_name(ontology):
    digital_reference_classes_and_names = []
    for index, (entity, name) in enumerate(getClassName(getClasses(ontology))):
        digital_reference_classes_and_names.append((entity, getSplittedText(name)))
    return digital_reference_classes_and_names


def get_ontology_triples(ontology, classes):
    triples = list(ontology.get_triples(None, None, None))
    for triple in triples:
        for el in triple:
            print(ontology._unabbreviate(el))
    return triples


def attachDocumentsToClasses(infineonBERT, listOfTuples):

    for index, (classObj, className) in enumerate(listOfTuples):
        text = getSplittedText(className)
        print("{} - Working on class {}".format(index, text))
        list_query_embeds = service.convert_query_to_embedding_Sentence_BERT(infineonBERT.encoder, [text])
        documents = service.multiprocess_search(text, list_query_embeds, infineonBERT.files_sentence_embeddings)
        for doc in documents:
            try:
                #print("\t\t" + str(doc['title']) + "\t\t" + str(doc['score'])) # mettere a posto l'encoding
                classObj.comment.append(doc['title'])
            except UnicodeEncodeError:
                #print("\t\t" + "**Impossible to print the title**")
                pass


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
    doc = nlp(query)
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


# import the Digital Reference
digitalReferenceOntology = importOntology()
print(digitalReferenceOntology.graph.dump())

# retrieve the classes of the digital reference
triples = digitalReferenceOntology.get_triples(None, None, None)
for triple in triples:
    for el in triple:
        digitalReferenceOntology._unabbreviate(el)

# get a list of tuples: (the class object, the names from the uri)
classTuple = getClassName(triples)

# instance of infineonBERT
infineonBERT = model.InfineonBERT("Infineon search engine")

start = time.time()
# attach documents to classes
attachDocumentsToClasses(infineonBERT, classTuple)


# print(lambda x: print(x.comment) for x in getClasses(digitalReferenceOntology))
print("Time needed to annotate all the classes: {}".format(time.time() - start))

# save the ontology to file with RDF/XML format
with open(PATH_SAVED_ONTOLOGY, "wb") as fp:
   digitalReferenceOntology.save(file=fp)

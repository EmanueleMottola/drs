import searchWithDR
import model
import search
import estastic
#import lsh

# return a listof dictionaries
# def reply_with_sentence_BERT(infineonBERT, text, engine):
def reply_with_sentence_BERT(drsBERT, text, sentences_embedings, id_vect):

    SEARCH_METHOD = 1

    list_query_embeds = drsBERT.convert_query_to_embedding([text])

    render_data = {}
    if SEARCH_METHOD == 1:
        render_data = search.multiprocess_search(list_query_embeds, sentences_embedings, id_vect)
    elif SEARCH_METHOD == 2:
        pass
        # render_data = estastic.search(client, list_query_embeds[0])
    elif SEARCH_METHOD == 3:
        pass
        # render_data = lsh.search_with_LSH(engine, list_query_embeds)
    else:
        raise Exception("No search model found!")

    return render_data

# returns a list of dictionaries, where every dictionary represents the results of search keyword
def reply_with_Digital_Reference(text, digital_reference_classes_and_names):
    keywords = searchWithDR.extract_keywords_from_query(text)
    annotations = []
    for keyword in keywords:
        annotation = searchWithDR.search_class_annotations(keyword, digital_reference_classes_and_names)
        if annotation['entity'] != "":
            annotations.append(annotation)
    #print(annotations)
    return annotations


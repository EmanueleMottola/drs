from flask import Flask, request, send_from_directory, make_response, render_template, current_app
#import sys
#sys.path.insert(1, './flask-api/')
import model, service
import os
import estastic
import searchWithDR
import annotateDR
import lsh
import logging
import time

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/query", methods=['POST'])
def collect_query():
    # text is retrieved from the form
    text = request.form['query']

    # the text is converted to vector
    start = time.time()
    render_data_SBERT = service.reply_with_sentence_BERT(infineonBERT, text, sentences_embeddings, id_vect)
    # render_data_SBERT = service.reply_with_sentence_BERT(infineonBERT, text, engine) # engine <> client
    # list_query_embeds = service.convert_query_to_embedding_Sentence_BERT(infineonBERT.encoder, [text])
    #print(np.shape(list_query_embeds[0]))
    #print(infineonBERT.files_sentence_embeddings[0]['embeddings'][1])
    middle = time.time()
    logging.debug(middle - start)
    render_data_DR = service.reply_with_Digital_Reference(text, digital_reference_classes_and_names)


    # search for similar vector representations

    # render_data = service.search(text, list_query_embeds, infineonBERT.files_sentence_embeddings)
    # render_data = service.multiprocess_search(text, list_query_embeds, infineonBERT.files_sentence_embeddings)
    #render_data = estastic.search(client, list_query_embeds[0])

    finish = time.time()

    logging.debug(finish-start)
    #print("Time to convert: " + str(middle-start))
    #print("Time to search: " + str(finish-middle))

    # create a json to return it as application/json
    render_data = {'sbert': render_data_SBERT, 'dr': render_data_DR}
    # print(render_data)
    result = {'results': render_data}
    #print(render_data_DR)
    return result


@app.route("/pdf/<filename>")
def get_pdf(filename):

    # file retrieved from the directory
    file = send_from_directory(app.config['UPLOAD_FOLDER'], filename + ".pdf")

    # response creation
    response = make_response(file)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=output.pdf'

    # response sent
    return response



if __name__ == "__main__":
    # load Sentence-BERT

    logging.debug("Sentence BERT loading..")
    infineonBERT = model.InfineonBERT("Infineon search engine")
    print("Sentence BERT loaded.")

    sentences_embeddings, id_vect = infineonBERT.files_sentence_embeddings

    # connect to Elastic-Search
    # client = estastic.start_client()
    #estastic.index_documents(client, infineonBERT.files_sentence_embeddings)

    # create a LSH index
    # engine = lsh.create_LSH_index(infineonBERT.files_sentence_embeddings)
    # engine = lsh.create_LSH_index_step()
    # logging.debug("Data indexed.")

    # laod the ontology
    onto = searchWithDR.importOntology(searchWithDR.PATH_SAVED_ONTOLOGY)
    digital_reference_classes_and_names = annotateDR.get_tuple_class_name(onto)
    logging.debug("Ontology loaded.")

    app.config['UPLOAD_FOLDER'] = 'C:/Users/Mottola/Documents/Thesis_local/text_corpus/pdf'
    app.run(debug=True, port=8080, use_reloader=False)
    logging.debug("Started server!")
from flask import Flask, request, send_from_directory, make_response, render_template, current_app
#import sys
#sys.path.insert(1, './flask-api/')
import model, service
import os
import estastic
import searchWithDR
import annotateDR
# import lsh
import logging
import time

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)


@app.route("/")
def hello():
    print("Fino qui tutto ok")
    return render_template("index.html")


@app.route("/query", methods=['POST'])
def collect_query():
    # text is retrieved from the form
    text = request.form['query']

    # the text is converted to vector
    start = time.time()
    render_data_SBERT = service.reply_with_sentence_BERT(drsBERT, text, sentences_embeddings, id_vect)

    middle = time.time()
    logging.debug(middle - start)

    # render_data_DR = service.reply_with_Digital_Reference(text, digital_reference_classes_and_names)
    render_data_DR = {}

    finish = time.time()

    logging.debug(finish-start)

    # create a json toy return it as application/json
    render_data = {'sbert': render_data_SBERT, 'dr': render_data_DR}

    result = {'results': render_data}
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
    drsBERT = model.drsBERT()
    print("Sentence BERT loaded.")

    #sentences_embeddings, id_vect = drsBERT.files_sentence_embeddings

    # connect to Elastic-Search
    # client = estastic.start_client()
    #estastic.index_documents(client, infineonBERT.files_sentence_embeddings)

    # create a LSH index
    # engine = lsh.create_LSH_index(infineonBERT.files_sentence_embeddings)
    # engine = lsh.create_LSH_index_step()
    # logging.debug("Data indexed.")

    # load the ontology
    # onto = searchWithDR.importOntology(searchWithDR.PATH_SAVED_ONTOLOGY)
    # digital_reference_classes_and_names = annotateDR.get_tuple_class_name(onto)
    # logging.debug("Ontology loaded.")

    app.config['UPLOAD_FOLDER'] = 'C:/Users/Mottola/Documents/Thesis_local/text_corpus/pdf'
    app.run(debug=True, port=8080, use_reloader=False)
    logging.debug("Started server!")


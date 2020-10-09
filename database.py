import psycopg2
import pickle
import logging
from tqdm import tqdm

PATH_TO_PDF = 'C:\\Users\\Mottola\\Documents\\Thesis_local\\text_corpus\\pdf\\'
PATH_TO_PICKLE_DATA = 'C:/Users/Mottola/Documents/InfineonSearchEngine/sentence_embeddings.pickle'
logging.basicConfig(level=logging.DEBUG)

def connect():
    conn = psycopg2.connect(host="localhost", database="cord-19", user="postgres", password="postgres")
    return conn

def create_tables():
    conn = connect()
    try:

        cur = conn.cursor()

        commands = (
            """
                CREATE TABLE documents (
                paper_id SERIAL PRIMARY KEY,
                title VARCHAR(65535) NOT NULL,
                document BYTEA
            )
            """,
            """
                CREATE TABLE sentences (
                sentence_id SERIAL PRIMARY KEY,
                sentence VARCHAR(65535) NOT NULL,
                embedding float[] NOT NULL,
                paper_id INTEGER NOT NULL,
                FOREIGN KEY (paper_id)
                        REFERENCES documents (paper_id)
                        ON UPDATE CASCADE ON DELETE CASCADE
            )
            """
        )

        for command in commands:
            cur.execute(command)

        conn.commit()

        logging.debug("Tables created.")

        cur.close()


    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(error)


    finally:

        conn.close()


def insert_data(sentence_embeddings):
    conn = connect()
    try:


        cur = conn.cursor()

        sql_documents = """
                INSERT INTO documents (title, hook, document)
                VALUES (%s, %s, %s)
                RETURNING doc_id;
                """

        sql_sentences = """
                INSERT INTO sentences (sentence, embedding, doc_id)
                VALUES (%s, %s, %s)
                RETURNING sentence_id;
                """

        sql_pickle = """
                INSERT INTO pickle (pickle_id, pickle_file)
                VALUES (%s, %s);
        """

        print("Inserting data..")

        for b in range(0, 10):
            with open("C:/Users/Mottola/Documents/InfineonSearchEngine/sentence_embeddings_" + str(b) + ".pickle", "rb") as pickle_file:
                p = pickle_file.read()
            cur.execute(sql_pickle, (b, p))

        for doc in sentence_embeddings:
            print("\t"+doc['title'])
            with open(PATH_TO_PDF + doc['hook'] + ".pdf", "rb") as f:
                pdf = f.read()
            cur.execute(sql_documents, (doc['title'], doc['hook'], pdf))
            doc_id = cur.fetchone()[0]
            # logging.debug(doc_id)

            for (sent, emb) in zip(doc['sentences'], doc['embeddings']):
                cur.execute(sql_sentences, (sent, emb.tolist(), doc_id))
                sentence_id = cur.fetchone()[0]
                # logging.debug(sentence_id)

        conn.commit()

        logging.debug("Data inserted.")


    except (Exception, psycopg2.DatabaseError) as error:

        logging.error(error)


    finally:

        conn.close()



def retrieve_embeddings_from_pickle():

    try:

        conn = connect()

        cur = conn.cursor()

        query = """
            SELECT pickle_file
            FROM pickle
            WHERE pickle_id = %s;
        """

        list_embeddings = []
        for index in range(10):
            logging.debug("Executing query {} ...".format(index))

            cur.execute(query, str(index))

            logging.debug("Executed!")
            list_embeddings += pickle.loads(cur.fetchone()[0])

        return list_embeddings

    except (Exception, psycopg2.DatabaseError) as error:

        logging.error(error)


    finally:

        conn.close()


def retrieve_sentence_and_doctitle(N):
    try:
        conn = connect()
        cur = conn.cursor()

        result = []

        for sentence_id in N:
            query = """
                    SELECT sentences.doc_id, sentence, hook
                    FROM documents, sentences
                    WHERE sentence_id = {} AND
                        sentences.doc_id = documents.doc_id;
                """
            cur.execute(query.format(sentence_id[1]))
            result.append(cur.fetchone())

        return result

    except (Exception, psycopg2.DatabaseError) as error:

        logging.error(error)


    finally:

        conn.close()


def retrieve_embeddings_from_sentences():

    try:
        conn = connect()
        cur = conn.cursor()

        result = []

        count_query = """
            SELECT COUNT(*) FROM sentences;
        """
        cur.execute(count_query)
        count = cur.fetchone()[0]

        for index in tqdm(range(1,count+1)):
            query = """
                    SELECT sentence_id, embedding
                    FROM sentences
                    WHERE sentence_id = {};
                """
            cur.execute(query.format(index))
            result.append(cur.fetchone())

        return result

    except (Exception, psycopg2.DatabaseError) as error:

        logging.error(error)


    finally:

        conn.close()


def retrieve_embedding(cur, index):

    try:
        query = """
                SELECT sentence_id, embedding
                FROM sentences
                WHERE sentence_id = {};
            """
        cur.execute(query.format(index))

        return cur.fetchone()

    except (Exception, psycopg2.DatabaseError) as error:

        logging.error(error)



def count_sentences():

    try:
        conn = connect()
        cur = conn.cursor()

        result = []

        count_query = """
            SELECT COUNT(*) FROM sentences;
        """
        cur.execute(count_query)
        count = cur.fetchone()[0]

        return count

    except (Exception, psycopg2.DatabaseError) as error:

        logging.error(error)


    finally:

        conn.close()






if __name__ == "__main__":

    with open(PATH_TO_PICKLE_DATA, 'rb') as f:
        files_s_embeddings = pickle.load(f)

    create_tables()
    insert_data(files_s_embeddings)



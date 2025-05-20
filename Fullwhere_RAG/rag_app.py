import os
import sys
import time
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from opensearchpy import (
    OpenSearch,
    RequestsHttpConnection,
    exceptions as opensearch_exceptions,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
INDEX_NAME = os.getenv("INDEX_NAME")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY non trouvé dans les variables d'environnement.")
if not OPENSEARCH_URL:
    raise ValueError("OPENSEARCH_URL non trouvé dans les variables d'environnement.")
if not INDEX_NAME:
    raise ValueError("INDEX_NAME non trouvé dans les variables d'environnement.")

DATA_PATH = "./data"

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY
)
llm = ChatOpenAI(
    model_name="gpt-4.1-2025-04-14", temperature=0, openai_api_key=OPENAI_API_KEY
)


def get_opensearch_client():
    """Crée et retourne un client OpenSearch."""
    client = OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=None,
        use_ssl=OPENSEARCH_URL.startswith("https"),
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
    )
    return client


def wait_for_opensearch(client, timeout=120):
    """Attend qu'OpenSearch soit disponible."""
    start_time = time.time()
    print("Attente d'OpenSearch...")
    while time.time() - start_time < timeout:
        try:
            if client.ping():
                print("OpenSearch est disponible.")
                return True
            time.sleep(5)
        except opensearch_exceptions.ConnectionError:
            print("Connexion à OpenSearch échouée, nouvelle tentative dans 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"Erreur inattendue lors de la connexion à OpenSearch: {e}")
            time.sleep(5)
    print("Timeout: OpenSearch n'est pas disponible après", timeout, "secondes.")
    return False


def create_index_if_not_exists(client):
    """Crée l'index avec le mapping k-NN si il n'existe pas."""
    if not client.indices.exists(index=INDEX_NAME):
        index_body = {
            "settings": {
                "index.knn": True,
                "index.knn.space_type": "cosinesimil",
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "metadata": {"type": "object", "enabled": False},
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": 3072,
                        "method": {
                            "name": "hnsw",
                            "space_type": "innerproduct",
                            "engine": "faiss",
                            "parameters": {"ef_construction": 128, "m": 24},
                        },
                    },
                }
            },
        }
        try:
            client.indices.create(index=INDEX_NAME, body=index_body)
            print(f"Index '{INDEX_NAME}' créé avec succès.")
        except opensearch_exceptions.RequestError as e:
            if e.error == "resource_already_exists_exception":
                print(f"Index '{INDEX_NAME}' existe déjà.")
            else:
                print(
                    f"Erreur lors de la création de l'index: {e.info['error']['root_cause']}"
                )
                raise
    else:
        print(f"Index '{INDEX_NAME}' existe déjà.")


def ingest_data():
    """Charge, découpe, vectorise et ingère les documents Markdown dans OpenSearch."""
    print("Démarrage de l'ingestion des données...")
    client = get_opensearch_client()
    if not wait_for_opensearch(client):
        sys.exit("Impossible de se connecter à OpenSearch. Arrêt de l'ingestion.")

    create_index_if_not_exists(client)

    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".md"):
            file_path = os.path.join(DATA_PATH, filename)
            print(f"Chargement de {file_path}...")
            loader = UnstructuredMarkdownLoader(file_path)
            documents.extend(loader.load())

    if not documents:
        print("Aucun document Markdown trouvé dans le dossier Data.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    print(f"Nombre de documents chargés: {len(documents)}")
    print(f"Nombre de chunks créés après découpage: {len(split_docs)}")

    opensearch_vector_store = OpenSearchVectorSearch(
        opensearch_url=OPENSEARCH_URL,
        index_name=INDEX_NAME,
        embedding_function=embeddings_model,
        http_auth=None,
        use_ssl=OPENSEARCH_URL.startswith("https"),
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        bulk_size=1000,
        timeout=60,
        engine="faiss",
    )

    print(f"Ingestion des chunks dans l'index '{INDEX_NAME}'...")
    opensearch_vector_store.add_documents(split_docs, bulk_size=1000)
    print("Ingestion terminée.")


def query_rag(user_query: str):
    """Interroge le RAG et affiche la réponse sourcée."""
    print(f"\nRequête: {user_query}")
    client = get_opensearch_client()
    if not wait_for_opensearch(client):
        sys.exit("Impossible de se connecter à OpenSearch. Arrêt de la requête.")

    if not client.indices.exists(index=INDEX_NAME):
        print(
            f"L'index '{INDEX_NAME}' n'existe pas. Veuillez d'abord ingérer des données."
        )
        print("Exécutez: docker compose run --rm python_app python rag_app.py ingest")
        return

    vector_store = OpenSearchVectorSearch(
        index_name=INDEX_NAME,
        embedding_function=embeddings_model,
        opensearch_url=OPENSEARCH_URL,
        http_auth=None,
        use_ssl=OPENSEARCH_URL.startswith("https"),
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    print("Génération de la réponse...")
    response = qa_chain.invoke({"query": user_query})

    print("\n--- Réponse ---")
    print(response.get("result", "Pas de réponse générée."))

    print("\n--- Sources ---")
    if response.get("source_documents"):
        for i, doc in enumerate(response["source_documents"]):
            print(f"Source {i+1}:")
            print(f"  Contenu: {doc.page_content[:200]}...")
            if "source" in doc.metadata:
                print(f"  Fichier: {doc.metadata['source']}")
            print("-" * 20)
    else:
        print("Aucun document source trouvé.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "ingest":
            ingest_data()
        elif command == "query":
            if len(sys.argv) > 2:
                query_text = " ".join(sys.argv[2:])
                query_rag(query_text)
            else:
                print("Mode requête interactif. Tapez 'exit' pour quitter.")
                while True:
                    user_q = input("Votre question: ")
                    if user_q.lower() == "exit":
                        break
                    if user_q.strip():
                        query_rag(user_q)
        else:
            print(f"Commande inconnue: {command}")
            print("Commandes disponibles: ingest, query [votre question]")
    else:
        print("Veuillez spécifier une commande: 'ingest' ou 'query [votre question]'")
        print(
            "Exemple pour interroger: python rag_app.py query Quelle est la recette de la quiche?"
        )
        print("Exemple pour ingérer: python rag_app.py ingest")

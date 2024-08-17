from src.data_converter import convert_data
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import os

def ingest_data(status):
    load_dotenv()

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    ASTRA_API_KEY = os.getenv("ASTRA_API_KEY")
    DB_ENDPOINT = os.getenv("DB_ENDPOINT")
    DB_ID = os.getenv("DB_ID")
    LANGCHAIN_TRACING_V2= True
    LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
    LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT="Law-Bot"

    gemini_embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    vector_store = AstraDBVectorStore(embedding = gemini_embedding,
                                      api_endpoint = DB_ENDPOINT,
                                      namespace = "constitution",
                                      token = ASTRA_API_KEY,
                                      collection_name = "Law_bot")
    is_full = status
    if is_full == None:#THIS MEANS THERE IS NO VECTORS CREATED IN DB
        text_chunks = convert_data()
        inserted_ids = vector_store.add_documents(text_chunks)
    else:
        return vector_store
    
    
    return vector_store,inserted_ids


if __name__ == "__main__":
    vector_store = ingest("done")#"done" HERE BECAUSE I HAD ALREADY CREATED DB IN experiments.ipynb
    print("DB has been initialized")

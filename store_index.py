from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as LC_Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


# Initialize Pinecone (serverless)
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "medicalchatbot"
# Connect to your index using host
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# import os
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

vectorstore = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("Data uploaded successfully with new SDK!")
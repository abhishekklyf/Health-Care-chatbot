import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from llama_index.core import SimpleDirectoryReader

# Load embeddings model
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
print(embeddings)

# Load documents from the 'data' directory
documents1 = SimpleDirectoryReader('data').load_data()

# Define a wrapper class to conform to the expected input structure for Qdrant
class DocumentWrapper:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)

# Split documents into chunks and wrap them
texts = []
for document in documents1:
    if isinstance(document, str):
        chunks = text_splitter.split_text(document)
    else:
        content = document.get_content() if hasattr(document, 'get_content') else str(document)
        chunks = text_splitter.split_text(content)
    for chunk in chunks:
        texts.append(DocumentWrapper(chunk))

# Print the second text chunk for verification
print(texts[1].page_content)

# Specify URL of Qdrant
url = "http://localhost:6333"

# Create Qdrant vector store from documents
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="vector_db"
)

print("Vector DB Successfully Created!")

import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import VectorStoreRetriever

# Initialize embeddings
# We assume a separate deployment for embeddings or standard one.
# For now, defaulting to a common name, user might need to adjust.
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
)

def get_retriever() -> VectorStoreRetriever:
    """
    Initializes and returns a ChromaDB retriever for the employee handbook.
    If the vector store exists, it loads it. Otherwise, it creates it.
    """
    persist_directory = "./chroma_db"
    
    # Path to the data
    file_path = "data/employee_handbook.txt"
    
    # Check if we assume it's already built or build on fly. 
    # For this demo, we will rebuild if db doesn't exist or just build in memory/persist.
    # To keep it robust, let's load and inspect.
    
    loader = TextLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    
    # Create the vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="employee_handbook"
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 2})

# Blog alignment: alias function
def ingest_docs(folder: str) -> None:
    """
    Ingests documents from a folder into the vector store.
    Matches the blog post example signature.
    """
    persist_directory = "./chroma_db"
    
    # Simple recursive loader for the folder
    # Note: In the real app we might use DirectoryLoader, but here we just replicate the logic
    # using our existing setup for specific file or folder.
    print(f"Ingesting docs from {folder}...")
    loader = TextLoader(os.path.join(folder, "employee_handbook.txt")) # Simplified for demo
    documents = loader.load()
    
    text_splitters = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitters.split_documents(documents)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="employee_handbook",
        persist_directory=persist_directory
    )
    # vectorstore.persist() # Chroma 0.4+ persists automatically or via specific calls depending on backend
    print("Ingestion complete.")

def query_handbook(query: str) -> str:
    """Helper to test the retriever"""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

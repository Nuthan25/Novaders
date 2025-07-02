import os
from typing import List, Dict, Any

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from langchain.schema import Document

import constants

class KnowledgeBase:
    def __init__(self, documents_dir: str = "knowledge_documents"):
        """Initialize the knowledge base with a directory for documents."""
        self.documents_dir = documents_dir
        self.embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
            region_name=constants.region
        )
        self.vector_store = None

        # Create documents directory if it doesn't exist
        if not os.path.exists(documents_dir):
            os.makedirs(documents_dir)

    def load_documents(self) -> List[Document]:
        """Load documents from the documents directory."""
        documents = []

        # Load text files
        if os.path.exists(os.path.join(self.documents_dir, "text")):
            text_loader = DirectoryLoader(
                os.path.join(self.documents_dir, "text"),
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents.extend(text_loader.load())

        # Load PDF files
        if os.path.exists(os.path.join(self.documents_dir, "pdf")):
            pdf_loader = DirectoryLoader(
                os.path.join(self.documents_dir, "pdf"),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            documents.extend(pdf_loader.load())

        # Load CSV files
        if os.path.exists(os.path.join(self.documents_dir, "csv")):
            csv_loader = DirectoryLoader(
                os.path.join(self.documents_dir, "csv"),
                glob="**/*.csv",
                loader_cls=CSVLoader
            )
            documents.extend(csv_loader.load())

        # Load Excel files
        if os.path.exists(os.path.join(self.documents_dir, "excel")):
            excel_loader = DirectoryLoader(
                os.path.join(self.documents_dir, "excel"),
                glob="**/*.xlsx",
                loader_cls=UnstructuredExcelLoader
            )
            documents.extend(excel_loader.load())

        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents by splitting them into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)

    def initialize_vector_store(self, documents: List[Document] = None) -> None:
        """Initialize the vector store with documents."""
        if documents is None:
            documents = self.load_documents()

        processed_docs = self.process_documents(documents)

        if processed_docs:
            self.vector_store = FAISS.from_documents(
                processed_docs,
                self.embeddings
            )
            # Save the vector store
            self.vector_store.save_local(os.path.join(self.documents_dir, "faiss_index"))
            print(f"Vector store initialized with {len(processed_docs)} document chunks.")
        else:
            print("No documents found to initialize vector store.")

    def load_vector_store(self) -> bool:
        """Load the vector store from disk if it exists."""
        index_path = os.path.join(self.documents_dir, "faiss_index")
        if os.path.exists(index_path):
            self.vector_store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
            return True
        return False

    def add_document(self, file_path: str) -> bool:
        """Add a single document to the knowledge base."""
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False

        file_ext = os.path.splitext(file_path)[1].lower()
        document = None

        try:
            if file_ext == ".txt":
                loader = TextLoader(file_path)
                document = loader.load()
            elif file_ext == ".pdf":
                loader = PyPDFLoader(file_path)
                document = loader.load()
            elif file_ext == ".csv":
                loader = CSVLoader(file_path)
                document = loader.load()
            elif file_ext in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(file_path)
                document = loader.load()
            else:
                print(f"Unsupported file type: {file_ext}")
                return False

            if document:
                processed_docs = self.process_documents(document)

                if self.vector_store is None:
                    # Initialize if not already done
                    self.initialize_vector_store(processed_docs)
                else:
                    # Add to existing vector store
                    self.vector_store.add_documents(processed_docs)
                    # Save the updated vector store
                    self.vector_store.save_local(os.path.join(self.documents_dir, "faiss_index"))

                print(f"Added {len(processed_docs)} document chunks to the knowledge base.")
                return True
        except Exception as e:
            print(f"Error adding document: {e}")
            return False

    def delete_document(self, condition: Dict[str, Any]) -> bool:
        """
        Delete specific documents from the knowledge base based on a condition.
        :param condition: A dictionary specifying the condition to match. For example,
                          {'metadata': {'author': 'John Doe'}}, or
                          {'content_contains': 'specific keyword'}.
        :return: True if deletion is successful, otherwise False.
        """
        if self.vector_store is None:
            if not self.load_vector_store():
                print("Vector store not initialized. No documents to delete.")
                return False

        try:
            filtered_documents = []
            for doc in self.vector_store.index_to_vector:
                match = False
                if 'metadata' in condition:
                    match = all(key in doc.metadata and doc.metadata[key] == value
                                for key, value in condition['metadata'].items())
                if 'content_contains' in condition:
                    match = match or (condition['content_contains'] in doc.page_content)

                if not match:
                    filtered_documents.append(doc)

            # Rebuild vector store with filtered documents
            self.vector_store = FAISS.from_documents(filtered_documents, self.embeddings)
            self.vector_store.save_local(os.path.join(self.documents_dir, "faiss_index"))
            print("Specified documents have been deleted.")
            return True
        except Exception as e:
            print(f"Error during deletion: {e}")
            return False

    def clear_all_data(self) -> bool:
        """
        Delete all data from the knowledge base, including documents and FAISS index.
        :return: True if successful, otherwise False.
        """
        try:
            # Remove all files in the documents directory
            for root, dirs, files in os.walk(self.documents_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            # Remove FAISS index
            index_path = os.path.join(self.documents_dir, "faiss_index")
            if os.path.exists(index_path):
                os.remove(index_path)

            self.vector_store = None
            print("All data has been cleared from the knowledge base.")
            return True
        except Exception as e:
            print(f"Error clearing data: {e}")
            return False

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the knowledge base for relevant documents."""
        if self.vector_store is None:
            if not self.load_vector_store():
                print("Vector store not initialized. Please add documents first.")
                return []

        results = self.vector_store.similarity_search_with_score(query, k=top_k)

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score
            })

        return formatted_results

# Initialize the knowledge base
kb = KnowledgeBase()
import pandas as pd
from langchain.tools import Tool
from knowledge_base import kb


def read_xlsx(path: str):
    try:
        path = path.strip()
        return pd.read_excel(rf'{path}', dtype=object).sample(5)
    except Exception as e:
        return f"Error reading xlsx file: {e}"


def read_csv(path: str):
    try:
        path = path.strip()
        return pd.read_csv(rf'{path}', dtype=object).sample(5)
    except Exception as e:
        return f"Error reading csv file: {e}"


def query_knowledge_base(query: str):
    """Query the knowledge base for information."""
    try:
        results = kb.query(query)
        if not results:
            return "No relevant information found in the knowledge base."

        response = "Here's what I found in the knowledge base:\n\n"
        for i, result in enumerate(results, 1):
            response += f"Document {i}:\n"
            response += f"Content: {result['content']}\n"
            if result['metadata']:
                response += f"Source: {result['metadata'].get('source', 'Unknown')}\n"
            response += "\n"

        return response
    except Exception as e:
        return f"Error querying knowledge base: {e}"


def add_to_knowledge_base(file_path: str):
    """Add a document to the knowledge base."""
    try:
        success = kb.add_document(file_path)
        if success:
            return f"Successfully added {file_path} to the knowledge base."
        else:
            return f"Failed to add {file_path} to the knowledge base."
    except Exception as e:
        return f"Error adding to knowledge base: {e}"

def delete_from_knowledge_base(document_id: str):
    """Delete a specific document from the knowledge base by its ID."""
    try:
        success = kb.delete_document(document_id)
        if success:
            return f"Successfully deleted document with ID {document_id} from the knowledge base."
        else:
            return f"Failed to delete document with ID {document_id} from the knowledge base."
    except Exception as e:
        return f"Error deleting from knowledge base: {e}"


def clear_knowledge_base():
    """Delete all documents from the knowledge base."""
    try:
        success = kb.clear_all_data()
        if success:
            return "Successfully cleared the entire knowledge base."
        else:
            return "Failed to clear the knowledge base."
    except Exception as e:
        return f"Error clearing the knowledge base: {e}"


tools = [
    Tool(
        name="Read .xlsx",
        func=read_xlsx,
        description="Useful for reading .xlsx file and returns sample data. Input should be a valid path."
    ),
    Tool(
        name="Read .csv",
        func=read_csv,
        description="Useful for reading .csv file and returns sample data. Input should be a valid path."
    ),
    Tool(
        name="Query Knowledge Base",
        func=query_knowledge_base,
        description="Search the knowledge base for information related to a query. Input should be a question or search term."
    ),
    Tool(
        name="Add to Knowledge Base",
        func=add_to_knowledge_base,
        description="Add a document to the knowledge base. Input should be a valid file path to a .txt, .pdf, .csv, or .xlsx file."
    ),
    Tool(
        name="Delete from Knowledge Base",
        func=delete_from_knowledge_base,
        description="Deletes a specific document from the knowledge base by its ID. Input should be a valid document ID."
    ),
    Tool(
        name="Clear Knowledge Base",
        func=clear_knowledge_base,
        description="Deletes all documents from the knowledge base."
    )
]

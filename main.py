from fastapi import FastAPI
from pydantic import BaseModel
import os
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

app = FastAPI()

class Consulta(BaseModel):
    pregunta: str

@app.post("/consultar")
async def consultar(data: Consulta):
    try:
        documents = []
        folder_path = "./documentos"  # Carpeta fija dentro del proyecto

        pdf_loader = PDFReader()
        docx_loader = DocxReader()

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.endswith(".pdf"):
                documents.extend(pdf_loader.load_data(file_path))
            elif filename.endswith(".docx"):
                documents.extend(docx_loader.load_data(file_path))

        embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        Settings.embed_model = embed_model
        Settings.llm = None

        index = VectorStoreIndex.from_documents(documents)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        query_engine = RetrieverQueryEngine(retriever=retriever)

        response = query_engine.query(data.pregunta)
        return {"respuesta": str(response)}

    except Exception as e:
        return {"error": str(e)}

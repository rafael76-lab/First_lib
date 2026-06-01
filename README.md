# RAG_I_Q_Pipeline

Librería de Python con dos clases para construir un pipeline RAG (Retrieval-Augmented Generation): una para inyectar documentos en una base de datos vectorial y otra para recuperar los documentos más relevantes dado un query.

## Clases

**Injection** — indexa tu documento una sola vez:
1. Carga un PDF
2. Lo divide en chunks de 500 caracteres con overlap de 50
3. Convierte cada chunk en un vector embedding usando `all-MiniLM-L6-v2`
4. Guarda los embeddings en ChromaDB

**Retrieval** — recupera documentos relevantes:
1. Carga la base de datos vectorial existente
2. Convierte el query en un embedding
3. Devuelve los chunks más similares mediante cosine similarity

## Instalación

```bash
pip install -e .
```

## Uso

```python
from RAG_I_Q_Pipeline import Injection, Retrieval

# Inyectar documento — solo una vez
injection = Injection("tu_documento.pdf", "carpeta donde se va a ubicar la base vectorial")
injection.RAG()

# Recuperar chunks relevantes
retrieval = Retrieval("carpeta de la base vectorial")
docs = retrieval.retrieve("Tu pregunta aquí")

for doc in docs:
    print(doc.page_content)
```

## Stack tecnológico
- **LangChain** — orquestación del pipeline
- **ChromaDB** — base de datos vectorial
- **HuggingFace** `all-MiniLM-L6-v2` (Microsoft) — embeddings
- **pypdf** — carga de PDFs
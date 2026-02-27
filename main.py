import os
import shutil
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

load_dotenv()

app = FastAPI(title="RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory session store: session_id -> chain
session_store: dict = {}

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_document(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".docx":
        return Docx2txtLoader(file_path).load()
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def build_chain_from_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    parser = StrOutputParser()

    prompt = PromptTemplate(
        template="""
        You are an AI helper that will answer the questions asked by the users from the context.
        If they ask for MCQ, Answer the Following, True or False, Summary, Fill-in-the-blanks,
        Translation in different language etc.
        {context}
        {question}
        """,
        input_variables=["context", "question"],
    )

    def format_docs(retrieved_chunks):
        return "\n\n".join(doc.page_content for doc in retrieved_chunks)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    return parallel_chain | prompt | llm | parser


# ─── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    session_id: str
    question: str

class QueryResponse(BaseModel):
    answer: str

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    message: str


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "RAG API is running 🚀"}


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[-1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    session_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{session_id}{ext}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        docs = load_document(save_path)
        chain = build_chain_from_docs(docs)
        session_store[session_id] = chain
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        message=f"✅ '{file.filename}' uploaded and processed successfully!"
    )


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chain = session_store.get(request.session_id)
    if not chain:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a document first."
        )

    try:
        answer = chain.invoke(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in session_store:
        del session_store[session_id]
        for ext in SUPPORTED_EXTENSIONS:
            path = os.path.join(UPLOAD_DIR, f"{session_id}{ext}")
            if os.path.exists(path):
                os.remove(path)
        return {"message": "Session cleared."}
    raise HTTPException(status_code=404, detail="Session not found.")


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
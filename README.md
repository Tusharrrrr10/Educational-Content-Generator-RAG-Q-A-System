# 📚 Educational Content Generator

A Retrieval-Augmented Generation (RAG) API that lets users upload documents and instantly generate educational content — MCQs, true/false questions, fill-in-the-blanks, summaries, and more — powered by GPT-4o-mini.

---

## 📌 Overview

Users upload a document (PDF, DOCX, or TXT), and the system processes it into a searchable vector store. They can then ask the system to generate any type of educational content based strictly on the document's content. Each upload gets its own isolated session so multiple users can work simultaneously without interference.

---

## 🧠 How It Works

```
User Uploads Document (PDF / DOCX / TXT)
            ↓
Load & Split into Chunks
            ↓
Generate Embeddings (OpenAI Embeddings)
            ↓
Store in FAISS Vector Store (per session)
            ↓
User sends a question / content request
            ↓
Retrieve Relevant Chunks → GPT-4o-mini generates content
            ↓
Return: MCQs / True-False / Fill-in-the-blanks / Summary etc.
```

---

## 🛠️ Tech Stack

| Component        | Tool / Library            |
|------------------|---------------------------|
| Language         | Python                    |
| Framework        | FastAPI                   |
| LLM              | OpenAI GPT-4o-mini        |
| Embeddings       | OpenAI Embeddings         |
| Vector Store     | FAISS                     |
| RAG Framework    | LangChain                 |
| PDF Loader       | PyPDFLoader               |
| DOCX Loader      | Docx2txtLoader            |
| Config           | python-dotenv             |
| Server           | Uvicorn                   |

---

## 📁 Project Structure

```
educational-content-generator/
│
├── main.py              # FastAPI app — routes, session management, RAG chain
├── uploaded_files/      # Temporarily stores uploaded documents
├── .env                 # API keys (not committed)
├── requirements.txt     # Dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/educational-content-generator.git
cd educational-content-generator
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📦 Requirements

```txt
fastapi
uvicorn
langchain
langchain-openai
langchain-community
faiss-cpu
pypdf
docx2txt
python-dotenv
pydantic
```

---

## 🔌 API Endpoints

### `GET /`
Health check — confirms the API is running.

---

### `POST /upload`
Upload a document to create a new session.

**Request:** `multipart/form-data`
| Field  | Type   | Description                        |
|--------|--------|------------------------------------|
| `file` | File   | PDF, DOCX, or TXT file to upload   |

**Response:**
```json
{
  "session_id": "uuid-string",
  "filename": "document.pdf",
  "message": "✅ 'document.pdf' uploaded and processed successfully!"
}
```

---

### `POST /ask`
Ask a question or request educational content from an uploaded document.

**Request body:**
```json
{
  "session_id": "your-session-id",
  "question": "Generate 5 MCQs from this document"
}
```

**Response:**
```json
{
  "answer": "1. Question...\n   a) ...\n   b) ...\n   ..."
}
```

---

### `DELETE /session/{session_id}`
Clear a session and delete the associated file.

**Response:**
```json
{
  "message": "Session cleared."
}
```

---

## 💡 Example Prompts

Once you have a `session_id`, try these in the `/ask` endpoint:

```
"Generate 5 multiple choice questions from this document"
"Create 10 true or false questions"
"Give me 5 fill-in-the-blank questions"
"Summarize this document in 5 bullet points"
"Generate a question and answer set for revision"
"Translate the summary into Hindi"
```

---

## 🔒 Notes

- Supported file types: `.pdf`, `.docx`, `.txt`
- Each upload generates a unique `session_id` — store it to query your document
- Sessions are stored in memory; they reset when the server restarts
- Your OpenAI API key is never hardcoded — always use `.env`

---

## 🙋‍♂️ Author

**Tushar Mishra**  
tusharmish25@gmail.com

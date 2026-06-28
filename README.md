# 💬 Talk to PDF

Chat with your PDF documents **100% locally** — no API keys, no cloud, no data leaving your machine.

Talk to PDF is a small [Streamlit](https://streamlit.io) app that turns any PDF into a question-answering chatbot. It runs a complete **Retrieval-Augmented Generation (RAG)** pipeline on your own computer using [Ollama](https://ollama.com) for the language and embedding models, and [ChromaDB](https://www.trychroma.com) as the local vector store.

> Upload a PDF → ask questions in plain English → get answers grounded in the document, with the exact source passages shown.

---

## ✨ Features

- 🔒 **Fully local & private** — everything runs through Ollama on your machine. Nothing is sent to any external service.
- 🧠 **Pick your own models** — choose any chat model and embedding model you have installed in Ollama, straight from the sidebar.
- 📄 **Smart PDF chunking** — text is split into overlapping chunks that try to break on sentence boundaries for cleaner retrieval.
- 🔍 **Source-grounded answers** — every answer comes with an expandable "Source Passages" view so you can verify where it came from.
- 🗑️ **Safe document removal** — a confirmation dialog prevents accidental clearing, and removing a document wipes its vector data cleanly.
- 🚫 **No accidental re-runs** — answers are cached per question, so reloading or removing the file won't re-trigger generation.

---

## 📦 Requirements

- **Python 3.12+**
- **[Ollama](https://ollama.com)** installed and running locally
- At least one **chat model** and (optionally) one **embedding model** pulled in Ollama

### Pull some models

```bash
# A small, capable chat model
ollama pull llama3.2

# (Optional) a dedicated embedding model — otherwise the built-in Chroma default is used
ollama pull nomic-embed-text
```

> If you don't pull an embedding model, the app falls back to **Chroma Default (local)**, which works out of the box.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/talk-to-pdf.git
cd talk-to-pdf
```

### 2. Install the dependencies

Using `pip` and a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or, if you use [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

### 3. Make sure Ollama is running

```bash
ollama serve        # if it isn't already running in the background
```

### 4. Launch the app

```bash
streamlit run main.py
```

Streamlit will open the app in your browser (usually at <http://localhost:8501>).

---

## 🖱️ How to Use

1. **Choose your models** in the sidebar — an LLM (chat) model and an embedding model.
2. **Upload a PDF** (one at a time). The document is processed and chunked into the vector store.
3. Once a document is loaded, the **model choices lock** so your embeddings stay consistent.
4. **Ask a question** in the box. The answer appears below, with the source passages it was drawn from.
5. Use the **✖** button to clear your question and answer.
6. Click **🗑️ Remove document** (and confirm) to clear everything, unlock the models, and load a new PDF.

---

## 🧠 How It Works

```
            ┌───────────────┐     chunks      ┌──────────────┐
  PDF  ───▶ │  PyPDF2 read  │ ──────────────▶ │   ChromaDB   │
            │  + chunking   │  (embeddings)   │ vector store │
            └───────────────┘                 └──────┬───────┘
                                                     │ top-k relevant chunks
                                                     ▼
  Question ─────────────────────────────────▶ ┌──────────────┐
                                               │    Ollama    │ ──▶ Answer + sources
                                               │  (LLM call)  │
                                               └──────────────┘
```

1. **Extract** — text is pulled from the PDF with `PyPDF2`.
2. **Chunk** — the text is split into ~1000-character chunks with a 200-character overlap, breaking on sentence ends where possible.
3. **Embed & store** — each chunk is embedded (via your chosen Ollama embedding model or Chroma's default) and stored in a persistent local ChromaDB collection.
4. **Retrieve** — your question is embedded and the most relevant chunks are fetched.
5. **Generate** — those chunks are passed as context to your chosen Ollama chat model, which answers the question.

---

## ⚙️ Configuration

A few constants near the top of [`main.py`](main.py) can be tweaked:

| Constant | Default | Description |
| --- | --- | --- |
| `CHUNK_SIZE` | `1000` | Target characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama's OpenAI-compatible endpoint |
| `EMBEDDING_MODEL_KEYWORDS` | `embed, nomic, bge, minilm, mxbai` | Keywords used to detect embedding models in `ollama list` |

The maximum upload size shown in the UI comes from Streamlit's own `server.maxUploadSize` setting (**200 MB by default**). To change it, create a `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 500
```

---

## ❓ FAQ

**Can I upload multiple PDFs at once?**
No — Talk to PDF is intentionally **one document at a time**. Remove the current document to load a different one.

**What's the file size limit?**
Whatever Streamlit's `server.maxUploadSize` is set to (200 MB by default). The app always displays the real configured value.

**Does this use OpenAI / send my data anywhere?**
No. The `openai` Python package is used only as a convenient client for Ollama's local OpenAI-compatible API. All requests go to `localhost`.

**Where is my data stored?**
Embeddings are kept in a local `chroma_db/` folder created next to the app. Removing a document deletes its collection.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io) — UI
- [Ollama](https://ollama.com) — local LLM + embeddings
- [ChromaDB](https://www.trychroma.com) — local vector database
- [PyPDF2](https://pypdf2.readthedocs.io) — PDF text extraction
- [openai](https://github.com/openai/openai-python) — client for Ollama's OpenAI-compatible API

---

## 📁 Project Structure

```
talk_to_pdf/
├── main.py            # The Streamlit app (UI + RAG pipeline)
├── requirements.txt   # Pinned dependencies
├── pyproject.toml     # Project metadata
├── README.md          # You are here
└── chroma_db/         # Local vector store (auto-created, git-ignored)
```

---

## 📝 License

Released under the [MIT License](LICENSE). Feel free to use, modify, and share.

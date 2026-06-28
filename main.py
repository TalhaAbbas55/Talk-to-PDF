import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
import subprocess
import PyPDF2
import uuid

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Known embedding model names (used to separate chat models from embedding models)
EMBEDDING_MODEL_KEYWORDS = ["embed", "nomic", "bge", "minilm", "mxbai"]


def get_ollama_models():
    """Run `ollama list` and return the list of installed model names."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().split("\n")
        models = []
        # Skip the header line (NAME  ID  SIZE  MODIFIED)
        for line in lines[1:]:
            if line.strip():
                # First column is the model name
                name = line.split()[0]
                models.append(name)
        return models
    except FileNotFoundError:
        st.error("Ollama is not installed or not in PATH. Install it from ollama.com")
        return []
    except subprocess.CalledProcessError as e:
        st.error(f"Error running 'ollama list': {e.stderr}")
        return []
    except Exception as e:
        st.error(f"Unexpected error fetching Ollama models: {str(e)}")
        return []


def split_models(all_models):
    """Split models into chat models and embedding models based on name keywords."""
    chat_models = []
    embedding_models = []
    for m in all_models:
        lower = m.lower()
        if any(keyword in lower for keyword in EMBEDDING_MODEL_KEYWORDS):
            embedding_models.append(m)
        else:
            chat_models.append(m)
    return chat_models, embedding_models


class SimpleModelSelector:
    """Handle model selection using models from `ollama list`."""

    def __init__(self, chat_models, embedding_models):
        self.chat_models = chat_models
        self.embedding_models = embedding_models

    def select_models(self, disabled=False, selected_llm=None, selected_embedding=None):
        st.sidebar.title("📚 Model Selection")

        if disabled:
            st.sidebar.info(
                "🔒 Models are locked while a document is loaded.\n\n"
                "Remove the document to choose different models."
            )
        else:
            st.sidebar.caption("Pick a chat model and an embedding model, then upload a PDF.")

        # ---- Chat / LLM model ----
        if not self.chat_models:
            st.sidebar.warning("No chat models found. Pull one with: ollama pull llama3.2")
            llm = None
        else:
            llm_index = (
                self.chat_models.index(selected_llm)
                if selected_llm in self.chat_models
                else 0
            )
            llm = st.sidebar.radio(
                "Choose LLM Model:",
                options=self.chat_models,
                index=llm_index,
                disabled=disabled,
            )

        # ---- Embedding model ----
        # Embedding options: any detected embedding models + Chroma's built-in default
        embedding_options = self.embedding_models + ["chroma-default"]
        embedding_index = (
            embedding_options.index(selected_embedding)
            if selected_embedding in embedding_options
            else 0
        )
        embedding = st.sidebar.radio(
            "Choose Embedding Model:",
            options=embedding_options,
            index=embedding_index,
            format_func=lambda x: "Chroma Default (local)" if x == "chroma-default" else x,
            disabled=disabled,
        )

        return llm, embedding


class SimplePDFProcessor:
    """Handle PDF processing and chunking"""

    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, pdf_file):
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    def create_chunks(self, text, pdf_file):
        chunks = []
        text_len = len(text)
        start = 0

        while start < text_len:
            # End of this chunk
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end]

            # Try to break at the last sentence end, but only if it leaves
            # a reasonably sized chunk (avoids tiny/empty chunks)
            if end < text_len:
                last_period = chunk.rfind(".")
                if last_period > self.chunk_size // 2:
                    end = start + last_period + 1
                    chunk = text[start:end]

            chunk = chunk.strip()
            if chunk:
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": chunk,
                        "metadata": {"source": pdf_file.name},
                    }
                )

            # Always move forward. next_start guarantees progress even if
            # overlap would otherwise pull us backward.
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks


class SimpleRAGSystem:
    """Simple RAG implementation using Ollama only."""

    def __init__(self, embedding_model, llm_model):
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Initialize ChromaDB
        self.db = chromadb.PersistentClient(path="./chroma_db")

        # Setup embedding function based on model
        self.setup_embedding_function()

        # Setup LLM (always Ollama)
        self.llm = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

        # Get or create collection
        self.collection = self.setup_collection()

    def setup_embedding_function(self):
        try:
            if self.embedding_model == "chroma-default":
                self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            else:
                # Use the selected Ollama embedding model
                self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key="ollama",
                    api_base=OLLAMA_BASE_URL,
                    model_name=self.embedding_model,
                )
        except Exception as e:
            st.error(f"Error setting up embedding function: {str(e)}")
            raise e

    def collection_name(self):
        # Sanitize collection name (Chroma has naming restrictions)
        safe_name = self.embedding_model.replace(":", "_").replace("/", "_").replace(".", "_")
        return f"documents_{safe_name}"

    def setup_collection(self):
        collection_name = self.collection_name()

        try:
            try:
                collection = self.db.get_collection(
                    name=collection_name, embedding_function=self.embedding_fn
                )
                st.info(f"Using existing collection for {self.embedding_model} embeddings")
            except:
                collection = self.db.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"model": self.embedding_model},
                )
                st.success(f"Created new collection for {self.embedding_model} embeddings")

            return collection

        except Exception as e:
            st.error(f"Error setting up collection: {str(e)}")
            raise e

    def delete_collection(self):
        """Drop the current collection so a removed document leaves no data behind."""
        try:
            self.db.delete_collection(self.collection_name())
        except Exception:
            # Nothing to delete / already gone is fine.
            pass

    def add_documents(self, chunks):
        try:
            if not self.collection:
                self.collection = self.setup_collection()

            self.collection.add(
                ids=[chunk["id"] for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks],
            )
            return True
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False

    def query_documents(self, query, n_results=3):
        try:
            if not self.collection:
                raise ValueError("No collection available")

            results = self.collection.query(query_texts=[query], n_results=n_results)
            return results
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None

    def generate_response(self, query, context):
        try:
            prompt = f"""
            Based on the following context, please answer the question.
            If you can't find the answer in the context, say so, or I don't know.

            Context: {context}

            Question: {query}

            Answer:
            """

            response = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )

            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None

    def get_embedding_info(self):
        return {
            "name": "Chroma Default" if self.embedding_model == "chroma-default" else self.embedding_model,
            "model": self.embedding_model,
        }


def _clear_query(query_key):
    """Callback for the ✖ button: clear the question box and the shown answer."""
    st.session_state[query_key] = ""
    st.session_state.last_answer = None


def _remove_document():
    """Clear all document state and unlock the model selection."""
    rag = st.session_state.get("rag_system")
    if rag is not None:
        rag.delete_collection()
    st.session_state.processed_file = None
    st.session_state.last_answer = None
    st.session_state.rag_system = None
    # Bump the key so the file uploader and the question box reset to empty.
    st.session_state.uploader_key += 1


@st.dialog("Remove this document?")
def confirm_remove_dialog():
    st.write(
        f"Remove **{st.session_state.processed_file}**?\n\n"
        "This clears the document and any answer shown below. You'll then be able "
        "to choose models again and upload a new PDF."
    )
    col_yes, col_no = st.columns(2)
    if col_yes.button("✅ Yes, remove", use_container_width=True):
        _remove_document()
        st.rerun()
    if col_no.button("❌ No, keep it", use_container_width=True):
        st.rerun()


def main():
    st.set_page_config(page_title="Talk to PDF", page_icon="💬")
    st.title("💬 Talk to PDF")

    # Initialize session state
    if "processed_file" not in st.session_state:
        st.session_state.processed_file = None  # name of the single loaded PDF
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None  # {"query", "response", "sources"}
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    # A document is loaded -> models are locked.
    locked = st.session_state.processed_file is not None

    # Fetch installed Ollama models
    all_models = get_ollama_models()
    chat_models, embedding_models = split_models(all_models)

    # Model selection (disabled while a document is loaded)
    model_selector = SimpleModelSelector(chat_models, embedding_models)
    if locked and st.session_state.rag_system is not None:
        # While locked, reflect the models the loaded document was built with.
        selected_llm = st.session_state.rag_system.llm_model
        selected_embedding = st.session_state.rag_system.embedding_model
    else:
        selected_llm = None
        selected_embedding = None

    llm_model, embedding_model = model_selector.select_models(
        disabled=locked,
        selected_llm=selected_llm,
        selected_embedding=selected_embedding,
    )

    if not llm_model:
        st.info("👆 No LLM model available. Pull one with `ollama pull llama3.2` and refresh.")
        return

    # Build / rebuild the RAG system. While unlocked, follow the chosen models.
    # While locked, the existing system is kept as-is.
    try:
        if not locked:
            rag = st.session_state.rag_system
            if (
                rag is None
                or rag.embedding_model != embedding_model
                or rag.llm_model != llm_model
            ):
                st.session_state.rag_system = SimpleRAGSystem(embedding_model, llm_model)

        embedding_info = st.session_state.rag_system.get_embedding_info()
        st.sidebar.info(
            f"Current Models:\n"
            f"- LLM: {st.session_state.rag_system.llm_model}\n"
            f"- Embedding: {embedding_info['name']}"
        )
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return

    # ---- File upload (single PDF) ----
    max_mb = st.get_option("server.maxUploadSize")  # the real, configured limit

    if st.session_state.processed_file is None:
        st.subheader("1️⃣ Choose your models  →  2️⃣ Upload a PDF")
        pdf_file = st.file_uploader(
            "Upload a PDF",
            type="pdf",
            key=f"uploader_{st.session_state.uploader_key}",
            help=f"One PDF at a time. Maximum size: {max_mb} MB.",
        )
        st.caption(f"📄 One PDF at a time · maximum size {max_mb} MB")

        if pdf_file is not None:
            processor = SimplePDFProcessor()
            with st.spinner("Processing PDF..."):
                try:
                    text = processor.read_pdf(pdf_file)
                    chunks = processor.create_chunks(text, pdf_file)
                    if st.session_state.rag_system.add_documents(chunks):
                        st.session_state.processed_file = pdf_file.name
                        st.session_state.last_answer = None
                        st.rerun()  # flip to the "loaded" view and lock the models
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
    else:
        st.success(f"📄 Loaded document: **{st.session_state.processed_file}**")
        if st.button("🗑️ Remove document", type="secondary"):
            confirm_remove_dialog()

    # ---- Query interface ----
    if st.session_state.processed_file:
        st.markdown("---")
        st.subheader("🔍 Talk to your PDF")

        query_key = f"query_{st.session_state.uploader_key}"
        col_q, col_x = st.columns([0.9, 0.1], vertical_alignment="bottom")
        with col_q:
            query = st.text_input(
                "Ask a question:",
                key=query_key,
                label_visibility="collapsed",
                placeholder="Ask a question about your PDF…",
            )
        with col_x:
            st.button(
                "✖",
                key=f"clear_{st.session_state.uploader_key}",
                on_click=_clear_query,
                args=(query_key,),
                help="Clear the question and answer",
                use_container_width=True,
            )

        rag = st.session_state.rag_system
        last = st.session_state.last_answer

        # Only generate when the question actually changed. Unrelated reruns
        # (e.g. removing the upload widget's file) won't re-trigger generation.
        if query and (last is None or last.get("query") != query):
            with st.spinner("Generating response..."):
                results = rag.query_documents(query)
                if results and results.get("documents") and results["documents"][0]:
                    response = rag.generate_response(query, results["documents"][0])
                    if response:
                        st.session_state.last_answer = {
                            "query": query,
                            "response": response,
                            "sources": results["documents"][0],
                        }
                else:
                    st.warning("No relevant passages found for that question.")
            last = st.session_state.last_answer

        # Show the cached answer (if it matches the current question).
        if query and last and last.get("query") == query:
            st.markdown("### 📝 Answer:")
            st.write(last["response"])
            with st.expander("View Source Passages"):
                for idx, doc in enumerate(last["sources"], 1):
                    st.markdown(f"**Passage {idx}:**")
                    st.info(doc)
    else:
        st.info("👆 Choose your models in the sidebar, then upload a PDF to get started!")


if __name__ == "__main__":
    main()

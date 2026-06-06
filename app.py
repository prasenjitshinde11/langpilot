import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import time

load_dotenv()

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg:         #0d0f14;
    --surface:    #161921;
    --border:     #252a36;
    --accent:     #5b7cf6;
    --accent-dim: #3d56cc;
    --green:      #3ecf8e;
    --red:        #f66b6b;
    --text:       #e8eaf0;
    --muted:      #6b7280;
    --user-bg:    #1e2333;
    --ai-bg:      #131720;
    --font-head:  'Syne', sans-serif;
    --font-mono:  'DM Mono', monospace;
}

/* ── Global Reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: var(--font-head);
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Main container padding ── */
[data-testid="stAppViewContainer"] > .main .block-container {
    padding: 2rem 2.5rem 4rem 2.5rem;
    max-width: 860px;
    margin: 0 auto;
}

/* ── Page title area ── */
.page-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}

.header-icon {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, var(--accent), #8b5cf6);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}

.header-text h1 {
    font-size: 1.45rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
    color: var(--text);
}

.header-text p {
    font-size: 0.78rem;
    color: var(--muted);
    margin: 2px 0 0;
    font-family: var(--font-mono);
    font-weight: 300;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 20px;
    background: rgba(62, 207, 142, 0.12);
    color: var(--green);
    border: 1px solid rgba(62, 207, 142, 0.25);
    margin-left: auto;
}

.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Chat container ── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    margin-bottom: 2rem;
}

/* ── Message bubbles ── */
.msg-wrapper {
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.msg-wrapper.user { flex-direction: row-reverse; }

.avatar {
    width: 34px; height: 34px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}

.avatar.user-av { background: linear-gradient(135deg, var(--accent), #8b5cf6); }
.avatar.ai-av   { background: linear-gradient(135deg, #2d3a5c, #1e2745);
                  border: 1px solid var(--border); }

.bubble {
    max-width: 78%;
    padding: 14px 18px;
    border-radius: 16px;
    font-size: 0.9rem;
    line-height: 1.65;
    font-family: var(--font-head);
    font-weight: 400;
    word-break: break-word;
}

.bubble.user-bubble {
    background: var(--user-bg);
    border: 1px solid var(--border);
    border-top-right-radius: 4px;
    color: var(--text);
}

.bubble.ai-bubble {
    background: var(--ai-bg);
    border: 1px solid var(--border);
    border-top-left-radius: 4px;
    color: var(--text);
}

.bubble-meta {
    font-size: 0.68rem;
    font-family: var(--font-mono);
    color: var(--muted);
    margin-top: 6px;
    font-weight: 300;
}

/* ── Source context pills ── */
.sources-wrap {
    margin-top: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.source-pill {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 400;
    padding: 3px 9px;
    border-radius: 20px;
    background: rgba(91, 124, 246, 0.1);
    border: 1px solid rgba(91, 124, 246, 0.25);
    color: var(--accent);
}

/* ── Input area ── */
.input-wrapper {
    position: sticky;
    bottom: 0;
    padding: 1.25rem 0 0.5rem;
    background: linear-gradient(to top, var(--bg) 80%, transparent);
}

/* Streamlit text input override */
[data-testid="stTextInput"] > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: var(--font-head) !important;
    font-size: 0.9rem !important;
    padding: 14px 18px !important;
    caret-color: var(--accent);
    transition: border-color 0.2s;
}

[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(91,124,246,0.12) !important;
    outline: none !important;
}

[data-testid="stTextInput"] label { display: none; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #7b6cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-head) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.3px;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
}

.stButton > button:hover  { opacity: 0.88; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* ── Sidebar sections ── */
.sidebar-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 14px;
}

.sidebar-label {
    font-size: 0.68rem;
    font-family: var(--font-mono);
    font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

.sidebar-value {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text);
}

/* ── Selectbox / slider labels ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"]    label,
[data-testid="stNumberInput"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--muted);
}

.empty-state .big-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-state h3 { font-size: 1.1rem; font-weight: 700; color: var(--text); margin-bottom: 6px; }
.empty-state p  { font-size: 0.82rem; font-family: var(--font-mono); font-weight: 300; }

/* ── Spinner tweak ── */
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# ── Cached Resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_name: str, temperature: float):
    return ChatGroq(model=model_name, temperature=temperature, max_retries=2)


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


@st.cache_resource(show_spinner=False)
def load_vector_store(_embeddings):
    return Chroma(
        persist_directory="./chroma_langchain_db",
        embedding_function=_embeddings,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:4px 0 18px'>
        <div style='font-family:"Syne",sans-serif;font-size:1.05rem;font-weight:800;
                    color:#e8eaf0;letter-spacing:-0.3px'>⚙️ Configuration</div>
        <div style='font-size:0.72rem;font-family:"DM Mono",monospace;
                    color:#6b7280;font-weight:300;margin-top:3px'>Model & retrieval settings</div>
    </div>
    """, unsafe_allow_html=True)

    model_name = st.selectbox(
        "LLM Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0,
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    top_k       = st.slider("Retrieval Top-K", min_value=1, max_value=10, value=3, step=1)

    st.markdown("<hr style='margin:18px 0'>", unsafe_allow_html=True)

    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='sidebar-section'>
            <div class='sidebar-label'>Queries</div>
            <div class='sidebar-value'>{st.session_state.total_queries}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='sidebar-section'>
            <div class='sidebar-label'>Messages</div>
            <div class='sidebar-value'>{len(st.session_state.messages)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='margin:18px 0'>", unsafe_allow_html=True)

    if st.button("🗑️  Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

    st.markdown("""
    <div style='font-size:0.68rem;font-family:"DM Mono",monospace;color:#6b7280;
                font-weight:300;margin-top:18px;line-height:1.7'>
        <b style='color:#9ca3af'>Stack</b><br>
        LangChain · Groq · ChromaDB<br>
        HuggingFace Embeddings<br>
        <span style='color:#5b7cf6'>all-mpnet-base-v2</span>
    </div>
    """, unsafe_allow_html=True)


# ── Load Resources ────────────────────────────────────────────────────────────
with st.spinner("Loading models…"):
    embeddings    = load_embeddings()
    vector_store  = load_vector_store(embeddings)
    model         = load_model(model_name, temperature)


# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='page-header'>
    <div class='header-icon'>🧠</div>
    <div class='header-text'>
        <h1>RAG Assistant</h1>
        <p>Retrieval-Augmented Generation · {model_name}</p>
    </div>
    <span class='status-badge'><span class='status-dot'></span> Ready</span>
</div>
""", unsafe_allow_html=True)


# ── Chat History ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class='empty-state'>
        <div class='big-icon'>💬</div>
        <h3>No messages yet</h3>
        <p>Type a question below — answers are grounded<br>in your vector knowledge base.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role    = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])
        ts      = msg.get("timestamp", "")

        if role == "user":
            st.markdown(f"""
            <div class='msg-wrapper user'>
                <div class='avatar user-av'>👤</div>
                <div>
                    <div class='bubble user-bubble'>{content}</div>
                    <div class='bubble-meta' style='text-align:right'>{ts}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            source_pills = "".join(
                f"<span class='source-pill'>📄 Chunk {i+1}</span>"
                for i in range(len(sources))
            )
            st.markdown(f"""
            <div class='msg-wrapper'>
                <div class='avatar ai-av'>🤖</div>
                <div>
                    <div class='bubble ai-bubble'>{content}
                        {f"<div class='sources-wrap'>{source_pills}</div>" if sources else ""}
                    </div>
                    <div class='bubble-meta'>{ts} · {len(sources)} chunks retrieved</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Input Row ─────────────────────────────────────────────────────────────────
st.markdown("<div class='input-wrapper'>", unsafe_allow_html=True)
col_inp, col_btn = st.columns([6, 1])

with col_inp:
    user_input = st.text_input(
        "Query",
        placeholder="Ask anything from your knowledge base…",
        key="query_input",
        label_visibility="collapsed",
    )

with col_btn:
    send = st.button("Send →", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ── Query Handling ────────────────────────────────────────────────────────────
def run_query(query: str):
    ts = time.strftime("%H:%M")

    # Save user message
    st.session_state.messages.append({
        "role": "user", "content": query, "timestamp": ts
    })
    st.session_state.total_queries += 1

    # Retrieve context
    results = vector_store.similarity_search(query, k=top_k)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""Answer this question based on the context below. Be concise and factual.

Context:
{context}

Question: {query}

Answer:"""

    with st.spinner("Thinking…"):
        response = model.invoke(prompt)

    st.session_state.messages.append({
        "role":      "assistant",
        "content":   response.content,
        "sources":   results,
        "timestamp": time.strftime("%H:%M"),
    })
    st.rerun()


if send and user_input.strip():
    run_query(user_input.strip())
elif user_input and user_input.strip().endswith("\n"):
    run_query(user_input.strip())
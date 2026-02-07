import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv
import numpy as np
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure API - works with both local .env and Streamlit Cloud secrets
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in environment variables!")
    st.stop()

# Page configuration with SEO
st.set_page_config(
    page_title="ClawGPT - OpenClaw AI Documentation Assistant | Instant Answers",
    page_icon="assets/logo.png" if Path("assets/logo.png").exists() else "🦞",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/openclaw/openclaw',
        'Report a bug': 'https://github.com/openclaw/openclaw/issues',
        'About': """
        # ClawGPT - OpenClaw Documentation Assistant

        AI-powered assistant that provides instant, intelligent answers about OpenClaw features,
        installation, configuration, and troubleshooting.

        **Built by RunShift | Powered by OpenRouter AI**
        """
    }
)

# SEO Meta Tags and Custom Styles
st.markdown("""
<!-- SEO Meta Tags -->
<head>
    <meta name="description" content="ClawGPT - AI-powered documentation assistant for OpenClaw. Get instant, intelligent answers about OpenClaw installation, configuration, channels, and features.">
    <meta name="keywords" content="ClawGPT, OpenClaw, AI Assistant, Documentation, Chatbot, WhatsApp Bot, Telegram Bot, Discord Bot, AI Documentation, OpenClaw Help, OpenClaw Guide, RunShift">
    <meta name="author" content="RunShift">
    <meta name="robots" content="index, follow">
    <meta property="og:title" content="ClawGPT - OpenClaw AI Documentation Assistant">
    <meta property="og:description" content="Get instant, intelligent answers about OpenClaw features and documentation with our AI-powered assistant.">
    <meta property="og:type" content="website">
    <meta property="og:site_name" content="ClawGPT by RunShift">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="ClawGPT - OpenClaw AI Documentation Assistant">
    <meta name="twitter:description" content="AI-powered documentation assistant for OpenClaw. Instant answers about installation, configuration, and features.">
    <link rel="canonical" href="https://clawgpt.streamlit.app">
</head>

<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Root Variables */
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #4ECDC4;
        --accent-color: #FFE66D;
        --dark-bg: #1a1a2e;
        --card-bg: #16213e;
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
        --gradient-1: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        --gradient-2: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
    }

    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Center Header Container */
    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 2rem 0 1rem 0;
        width: 100%;
    }

    .logo-wrapper {
        display: flex;
        justify-content: center;
        width: 100%;
        margin-bottom: 1rem;
    }

    .brand-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -1px;
        text-align: center;
    }

    .tagline {
        font-size: 1.3rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        font-weight: 400;
        text-align: center;
    }

    .highlight {
        color: var(--secondary-color);
        font-weight: 600;
    }

    /* Stats Bar */
    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 3rem;
        padding: 1.5rem;
        margin: 1.5rem auto;
        max-width: 800px;
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.1) 0%, rgba(255, 107, 107, 0.1) 100%);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stat-item {
        text-align: center;
    }

    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary-color);
    }

    .stat-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Message Styles */
    .stChatMessage {
        border-radius: 16px !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }

    /* Button Styles */
    .stButton > button {
        border-radius: 12px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        margin: 2rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: var(--text-secondary);
        font-size: 0.85rem;
    }

    .footer a {
        color: var(--primary-color);
        text-decoration: none;
    }

    .footer a:hover {
        text-decoration: underline;
    }

    /* Buy Me Coffee Button */
    .coffee-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, #FF813F 0%, #FFD23F 100%);
        color: #000;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.3s ease;
        margin-top: 0.5rem;
    }

    .coffee-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 129, 63, 0.4);
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Responsive */
    @media (max-width: 768px) {
        .brand-title {
            font-size: 2.5rem;
        }
        .tagline {
            font-size: 1rem;
        }
        .stats-bar {
            flex-direction: column;
            gap: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_vector_store():
    """Load the pre-built vector store."""
    store_dir = Path(__file__).parent / "vector_store"

    if not store_dir.exists():
        return None, None, None

    # Load embeddings
    embeddings = np.load(store_dir / "embeddings.npy")

    # Load chunks and metadata
    with open(store_dir / "chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    return embeddings, data["chunks"], data["metadata"]


@st.cache_data(ttl=300)
def get_query_embedding(text: str):
    """Get embedding for a query using OpenRouter (cached for 5 min)."""
    response = requests.post(
        url="https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://clawgpt.streamlit.app",
            "X-Title": "ClawGPT",
        },
        json={
            "model": "google/gemini-embedding-001",
            "input": text,
            "encoding_format": "float"
        },
        timeout=30
    )

    if response.status_code == 200:
        result = response.json()
        return np.array(result["data"][0]["embedding"], dtype=np.float32)
    else:
        return None


def get_relevant_context_fast(embeddings: np.ndarray, chunks: list, metadata: list, query: str, k: int = 5):
    """Retrieve relevant document chunks using vectorized numpy operations."""
    if embeddings is None:
        return [], []

    # Get query embedding
    query_emb = get_query_embedding(query)

    if query_emb is None:
        return [], []

    # Vectorized cosine similarity - MUCH faster than loop
    # Normalize vectors
    query_norm = query_emb / np.linalg.norm(query_emb)
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / emb_norms

    # Compute all similarities at once
    similarities = np.dot(embeddings_normalized, query_norm)

    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:k]

    contexts = []
    sources = []

    for idx in top_indices:
        contexts.append(chunks[idx])
        sources.append({
            "source": metadata[idx]["source"],
            "title": metadata[idx]["title"],
            "score": round(float(similarities[idx]) * 100, 1)
        })

    return contexts, sources


def generate_response(query: str, contexts: list):
    """Generate response using OpenRouter with DeepSeek model."""

    context_str = "\n\n---\n\n".join(contexts)

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://clawgpt.streamlit.app",
            "X-Title": "ClawGPT",
        },
        json={
            "model": "tngtech/deepseek-r1t2-chimera:free",
            "messages": [
                {
                    "role": "system",
                    "content": """You are ClawGPT, an expert AI documentation assistant for OpenClaw.
Provide accurate, helpful answers based ONLY on the provided documentation context.
Use markdown formatting. Be concise but complete.
If info isn't in the context, say so honestly."""
                },
                {
                    "role": "user",
                    "content": f"DOCUMENTATION:\n{context_str}\n\nQUESTION: {query}"
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.3
        },
        timeout=60
    )

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}"


def render_header():
    """Render the centered header with logo."""
    logo_path = Path(__file__).parent / "assets" / "logo.png"

    # Check if logo exists and encode to base64 for HTML
    if logo_path.exists():
        import base64
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="width: 120px; height: auto; border-radius: 16px; margin-bottom: 1rem;">'
    else:
        logo_html = '<span style="font-size: 5rem;">🦞</span>'

    # Render fully centered header using HTML
    st.markdown(f'''
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 2rem 0 1rem 0; width: 100%;">
        {logo_html}
        <h1 class="brand-title">ClawGPT</h1>
        <p class="tagline">
            AI-Powered Documentation Assistant for <span class="highlight">OpenClaw</span>
        </p>
    </div>
    ''', unsafe_allow_html=True)


def render_stats():
    """Render the stats bar."""
    st.markdown("""
    <div class="stats-bar animate-fade-in">
        <div class="stat-item">
            <div class="stat-value">96+</div>
            <div class="stat-label">Docs</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">⚡</div>
            <div class="stat-label">Instant</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">24/7</div>
            <div class="stat-label">Available</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">🎯</div>
            <div class="stat-label">Accurate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        logo_path = Path(__file__).parent / "assets" / "logo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=60)

        st.markdown("### 🦞 ClawGPT")
        st.markdown("Your intelligent guide to OpenClaw documentation")

        st.markdown("---")

        st.markdown("#### ✨ What I Can Help With")
        features = [
            ("🚀", "Installation & Setup"),
            ("⚙️", "Configuration"),
            ("📱", "Channel Integrations"),
            ("🔧", "Troubleshooting"),
            ("🤖", "Skills & Automation"),
            ("💾", "Memory & Sessions"),
        ]

        for icon, feature in features:
            st.markdown(f"{icon} {feature}")

        st.markdown("---")

        st.markdown("#### ⚙️ Settings")
        num_results = st.slider(
            "Context chunks",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of document sections to retrieve"
        )
        show_sources = st.checkbox("Show sources", value=True)

        st.markdown("---")

        st.markdown("#### 🔗 Resources")
        st.markdown("""
- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
- [OpenClaw Docs](https://docs.openclaw.ai)
        """)

        st.markdown("---")

        st.markdown("#### 🚀 RunShift")
        st.markdown("""
<a href="https://runshift-hq.vercel.app/" target="_blank" style="color: #4ECDC4; text-decoration: none;">
    🌐 Visit RunShift
</a>
        """, unsafe_allow_html=True)

        st.markdown("""
<a href="https://buymeacoffee.com/runshift" target="_blank" class="coffee-btn" style="display: inline-flex; align-items: center; gap: 0.5rem; background: linear-gradient(135deg, #FF813F 0%, #FFD23F 100%); color: #000; padding: 0.5rem 1rem; border-radius: 10px; font-weight: 600; text-decoration: none; margin-top: 0.5rem;">
    ☕ Buy Me a Coffee
</a>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.75rem;">
            Built by <strong>RunShift</strong><br>
            Powered by OpenRouter
        </div>
        """, unsafe_allow_html=True)

    return num_results, show_sources


def render_example_questions():
    """Render example question cards."""
    st.markdown("### 💡 Popular Questions")

    examples = [
        ("🚀", "How do I install OpenClaw?"),
        ("📱", "How to set up WhatsApp?"),
        ("🔧", "Configuration options?"),
        ("💾", "How does memory work?"),
        ("🌐", "How to run on VPS?"),
        ("🔒", "Security best practices?"),
    ]

    cols = st.columns(2)
    for idx, (icon, question) in enumerate(examples):
        with cols[idx % 2]:
            if st.button(f"{icon} {question}", key=f"ex_{idx}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()


def main():
    # Render header
    render_header()

    # Render stats
    render_stats()

    # Render sidebar and get settings
    num_results, show_sources = render_sidebar()

    # Load vector store
    embeddings, chunks, metadata = load_vector_store()

    if embeddings is None:
        st.error("""
        ⚠️ **Vector database not found!**

        Please run: `python create_embeddings.py`
        """)
        return

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Check for pending question from example buttons
    pending_question = None
    if "pending_question" in st.session_state and st.session_state.pending_question:
        pending_question = st.session_state.pending_question
        st.session_state.pending_question = None

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🦞"):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and show_sources:
                with st.expander("📚 Sources", expanded=False):
                    for src in message["sources"]:
                        st.markdown(f"• **{src['title']}** - {src['score']}%")

    # Show example questions if no messages and no pending question
    if not st.session_state.messages and not pending_question:
        render_example_questions()

    # Get prompt from chat input or pending question
    prompt = st.chat_input("Ask anything about OpenClaw...")

    # Use pending question if available
    if pending_question:
        prompt = pending_question

    # Process the prompt
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🦞"):
            with st.spinner("🔍 Searching..."):
                contexts, sources = get_relevant_context_fast(embeddings, chunks, metadata, prompt, k=num_results)

                if not contexts:
                    response = "I couldn't find relevant info. Try rephrasing your question."
                    sources = []
                else:
                    response = generate_response(prompt, contexts)

                st.markdown(response)

                if show_sources and sources:
                    with st.expander("📚 Sources", expanded=False):
                        for src in sources:
                            st.markdown(f"• **{src['title']}** - {src['score']}%")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })

    # Footer
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        <p>🦞 <strong>ClawGPT</strong> - AI Documentation Assistant for OpenClaw</p>
        <p>
            Built with ❤️ by <a href="https://runshift-hq.vercel.app/" target="_blank">RunShift</a> |
            <a href="https://buymeacoffee.com/runshift" target="_blank">☕ Support Us</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

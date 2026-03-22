import streamlit as st
import sys
import os
import io
import base64
sys.path.insert(0, os.path.abspath("."))

from gtts import gTTS
import speech_recognition as sr
from src.pipeline.ingest_pipeline import ingest_pdf, ingest_url
from src.pipeline.query_pipeline import run_query
from src.retrieval.vector_store import _close_chroma_client

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chatbot")
st.caption("Ask questions by voice or text — powered by LLaMA 3.1 + Groq")

# ── TTS Helper ────────────────────────────────────────────
def text_to_speech(text: str) -> str:
    """Convert answer text to audio HTML player."""
    try:
        clean = text[:500].replace("\n", " ").strip()
        tts = gTTS(text=clean, lang="en", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return f"""
        <audio autoplay controls style="width:100%;margin-top:8px;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    except Exception as e:
        return f"<p style='color:red'>Audio error: {e}</p>"

# ── STT Helper ────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio bytes using Google Speech Recognition."""
    try:
        r = sr.Recognizer()
        audio_buf = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_buf) as source:
            audio = r.record(source)
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except Exception:
        return ""

# ── Session state defaults ────────────────────────────────
defaults = {
    "messages": [],
    "ingested": False,
    "source_name": "",
    "tts_enabled": False,
    "stt_enabled": False,
    "audio_processed": False,
    "pending_voice": None,
    "mic_key": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Load Your Data")
    source_type = st.radio("Source type", ["URL", "PDF"])

    # ── URL ingestion ──────────────────────────────────────
    if source_type == "URL":
        url_input = st.text_input(
            "Enter a URL",
            placeholder="https://en.wikipedia.org/wiki/..."
        )
        if st.button("🔄 Ingest URL", use_container_width=True):
            if url_input.strip():
                with st.spinner("Fetching and indexing..."):
                    try:
                        _close_chroma_client()
                        ingest_url(url_input.strip())
                        st.session_state.ingested = True
                        st.session_state.source_name = url_input.strip()
                        st.session_state.messages = []
                        st.session_state.audio_processed = False
                        st.session_state.pending_voice = None
                        st.session_state.mic_key += 1
                        st.success("✅ URL ingested!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            else:
                st.warning("Please enter a URL.")

    # ── PDF ingestion ──────────────────────────────────────
    else:
        pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if st.button("🔄 Ingest PDF", use_container_width=True):
            if pdf_file:
                with st.spinner("Reading and indexing PDF..."):
                    try:
                        save_path = f"data/raw/{pdf_file.name}"
                        os.makedirs("data/raw", exist_ok=True)
                        with open(save_path, "wb") as f:
                            f.write(pdf_file.read())
                        _close_chroma_client()
                        ingest_pdf(save_path)
                        st.session_state.ingested = True
                        st.session_state.source_name = pdf_file.name
                        st.session_state.messages = []
                        st.session_state.audio_processed = False
                        st.session_state.pending_voice = None
                        st.session_state.mic_key += 1
                        st.success(f"✅ {pdf_file.name} ingested!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            else:
                st.warning("Please upload a PDF.")

    # ── Active source ──────────────────────────────────────
    if st.session_state.ingested:
        st.divider()
        name = st.session_state.source_name
        st.success(
            f"📄 {name[:45]}..."
            if len(name) > 45
            else f"📄 {name}"
        )
        if st.button("🗑️ Clear & Reset", use_container_width=True):
            st.session_state.messages = []
            st.session_state.ingested = False
            st.session_state.source_name = ""
            st.session_state.audio_processed = False
            st.session_state.pending_voice = None
            st.session_state.mic_key += 1
            st.rerun()

    # ── Audio Settings ─────────────────────────────────────
    st.divider()
    st.subheader("🔊 Audio Settings")

    # TTS toggle
    st.session_state.tts_enabled = st.toggle(
        "🔊 Read answers aloud (TTS)",
        value=st.session_state.tts_enabled,
        help="Speaks the answer after each response using Google TTS"
    )

    # STT toggle — resets mic state when switched
    prev_stt = st.session_state.stt_enabled
    new_stt = st.toggle(
        "🎙️ Voice input (STT)",
        value=st.session_state.stt_enabled,
        help="Record your question using the browser mic"
    )

    # STT just turned ON → reset everything for fresh start
    if new_stt and not prev_stt:
        st.session_state.stt_enabled = True
        st.session_state.audio_processed = False
        st.session_state.pending_voice = None
        st.session_state.mic_key += 1
        st.rerun()

    # STT just turned OFF → clean up
    elif not new_stt and prev_stt:
        st.session_state.stt_enabled = False
        st.session_state.audio_processed = False
        st.session_state.pending_voice = None
        st.rerun()

    else:
        st.session_state.stt_enabled = new_stt

    if st.session_state.tts_enabled:
        st.caption("🔊 TTS ON — answers will be read aloud")
    if st.session_state.stt_enabled:
        st.caption("🎙️ STT ON — mic button appears below")

# ── Main area — landing page ──────────────────────────────
if not st.session_state.ingested:
    st.info("👈 Load a URL or PDF from the sidebar to start chatting!")
    st.markdown("### Try these example URLs:")
    for url in [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "https://en.wikipedia.org/wiki/Large_language_model",
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://huggingface.co/blog/rag",
    ]:
        st.code(url, language=None)
    st.stop()

# ── Chat history ──────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    st.caption(src)

# ── Voice input section ───────────────────────────────────
voice_question = None

if st.session_state.stt_enabled:
    st.markdown("---")
    st.markdown("🎙️ **Record your question:**")

    # mic_key changes every time → forces a completely fresh widget
    audio_input = st.audio_input(
        "Click the mic, speak, then click stop",
        key=f"mic_recorder_{st.session_state.mic_key}"
    )

    if audio_input is not None and not st.session_state.audio_processed:
        with st.spinner("🎙️ Transcribing..."):
            spoken = transcribe_audio(audio_input.read())
            if spoken:
                st.success(f"✅ Heard: **{spoken}**")
                st.session_state.audio_processed = True
                st.session_state.pending_voice = spoken
            else:
                st.warning("⚠️ Could not transcribe — please speak clearly and try again.")

    # After recording — show reset button for next question
    if st.session_state.get("audio_processed"):
        if st.button(
            "🎙️ Record new question",
            use_container_width=False,
            key="reset_mic_btn"
        ):
            st.session_state.audio_processed = False
            st.session_state.pending_voice = None
            st.session_state.mic_key += 1
            st.rerun()

    # Consume pending voice question
    if st.session_state.get("pending_voice"):
        voice_question = st.session_state.pending_voice
        st.session_state.pending_voice = None

# ── Chat input ────────────────────────────────────────────
typed = st.chat_input("Type your question here...")
prompt = voice_question or typed

if prompt:
    # ── User message ───────────────────────────────────────
    display_prompt = f"🎙️ *{prompt}*" if voice_question else prompt
    st.session_state.messages.append({
        "role": "user",
        "content": display_prompt
    })
    with st.chat_message("user"):
        st.markdown(display_prompt)

    # ── Assistant response ─────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = run_query(prompt)
                answer = result["answer"]
                sources = result["sources"]

                # Show answer
                st.markdown(answer)

                # Show sources
                if sources:
                    with st.expander("📄 Sources"):
                        for src in sources:
                            st.caption(src)

                # TTS — read answer aloud
                if st.session_state.tts_enabled:
                    with st.spinner("🔊 Generating audio..."):
                        audio_html = text_to_speech(answer)
                        st.components.v1.html(audio_html, height=70)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                err = f"❌ Error generating answer: {e}"
                st.error(err)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err,
                    "sources": []
                })
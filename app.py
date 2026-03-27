import streamlit as st
import tempfile
import speech_recognition as sr
from chatbot import workflow  

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🧠",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
.user-msg {
    background-color: #1e293b;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.bot-msg {
    background-color: #020617;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.title("🧠 Medical AI Assistant")
st.caption("RAG + LangGraph + Voice Enabled")

# -------------------- SESSION STATE --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.last_audio = None
        st.session_state.pending_question = None
        st.rerun()

# -------------------- CHAT DISPLAY --------------------
st.subheader("💬 Chat")

for chat in st.session_state.chat_history:
    st.markdown(f"<div class='user-msg'>🧑‍💻 {chat['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-msg'>🤖 {chat['bot']}</div>", unsafe_allow_html=True)

# -------------------- INPUT --------------------
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.chat_input("Ask a medical question...")

with col2:
    audio_file = st.audio_input("🎤")

# -------------------- VOICE INPUT FIX --------------------
if audio_file is not None:
    current_audio = audio_file.getvalue()

    # ✅ Process ONLY new audio
    if st.session_state.last_audio != current_audio:

        st.session_state.last_audio = current_audio

        with st.spinner("Transcribing... 🎙️"):
            recognizer = sr.Recognizer()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(current_audio)
                tmp_path = tmp.name

            with sr.AudioFile(tmp_path) as source:
                audio_data = recognizer.record(source)

            
            text = recognizer.recognize_google(audio_data)
            st.success(f"🗣️ You said: {text}")

            # ✅ Store for processing
            st.session_state.pending_question = text

            st.rerun()

        

# -------------------- PROCESS --------------------
# Priority: text input > voice input
final_query = None

if user_input:
    final_query = user_input

elif st.session_state.pending_question:
    final_query = st.session_state.pending_question
    st.session_state.pending_question = None  # reset after use

# -------------------- RUN WORKFLOW --------------------
if final_query:
    with st.spinner("Thinking... 🤔"):

        result = workflow.invoke({
            "question": final_query,
            "evaluate_dec": "",
            "answer": "",
            "general_gk": ""
        })

        answer = result["answer"]

        # Save chat
        st.session_state.chat_history.append({
            "user": final_query,
            "bot": answer
        })

        st.rerun()
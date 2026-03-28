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

/* Main background */
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

/* Title */
h1 {
    color: #38bdf8;
    text-align: center;
}

/* Chat container spacing */
.block-container {
    padding-top: 2rem;
}

/* User message */
.user-msg {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    padding: 12px;
    border-radius: 15px;
    margin: 10px 0;
    width: fit-content;
    max-width: 70%;
    margin-left: auto;
    font-size: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}

/* Bot message */
.bot-msg {
    background: linear-gradient(135deg, #10b981, #06b6d4);
    color: white;
    padding: 12px;
    border-radius: 15px;
    margin: 10px 0;
    width: fit-content;
    max-width: 70%;
    font-size: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
}

/* Input */
textarea {
    border-radius: 10px !important;
}

/* Buttons */
button {
    border-radius: 10px !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white !important;
}

/* Alerts */
.stAlert {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<h1>🧠 MedBrain</h1>
<p style='text-align:center; color:#94a3b8;'>
Ask anything. Heal Smarter 💊
</p>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("""
    🔹 **Model:** Local LLM  
    🔹 **Retriever:** FAISS  
    🔹 **Mode:** RAG + Agent  
    """)

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.last_audio = None
        st.session_state.pending_question = None
        st.rerun()

# -------------------- CHAT DISPLAY --------------------
st.markdown("---")
st.subheader("💬 Chat Interface")

for chat in st.session_state.chat_history:
    # User (right)
    st.markdown(f"""
    <div style="display:flex; justify-content:flex-end;">
        <div class='user-msg'>🧑‍💻 {chat['user']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Bot (left)
    st.markdown(f"""
    <div style="display:flex; justify-content:flex-start;">
        <div class='bot-msg'>🤖 {chat['bot']}</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------- INPUT --------------------
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.chat_input("Ask a medical question...")

with col2:
    audio_file = st.audio_input("🎤")

# -------------------- VOICE INPUT FIX --------------------
if audio_file is not None:
    current_audio = audio_file.getvalue()

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

            st.session_state.pending_question = text
            st.rerun()

      
            st.error("")

# -------------------- PROCESS --------------------
final_query = None

if user_input:
    final_query = user_input

elif st.session_state.pending_question:
    final_query = st.session_state.pending_question
    st.session_state.pending_question = None

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

        st.session_state.chat_history.append({
            "user": final_query,
            "bot": answer
        })

        st.rerun()
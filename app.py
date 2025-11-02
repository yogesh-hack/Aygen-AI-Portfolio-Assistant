import streamlit as st
import requests
import os
import time
from streamlit_chat import message

# ---------------- Configuration ----------------
API_URL = "http://127.0.0.1:8000/chat"
VOICE_BASE_URL = "http://127.0.0.1:8000/voice/"

st.set_page_config(page_title="🎙️ Aygen Voice Assistant", page_icon="🤖", layout="centered")

# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎙️ Aygen Voice Assistant")
st.markdown("💬 Ask anything related to Yogesh Sir's portfolio or general questions. Aygen replies in Hindi or English with voice!")

# ---------------- Chat Interface ----------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "message": user_input})

    with st.spinner("Aygen is thinking..."):
        # Format chat history correctly for FastAPI
        formatted_history = []
        for msg in st.session_state.messages[:-1]:  # Exclude the current message
            formatted_history.append({
                "role": msg["role"],
                "message": msg["message"]
            })

        payload = {
            "user_message": user_input,
            "chat_history": formatted_history
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=90)
            data = response.json()
            
            if "answer" in data:
                answer = data["answer"]
                print("✅ Answer received:", answer)
                audio_url = VOICE_BASE_URL + os.path.basename(data["audio_url"])
                st.session_state.messages.append({
                    "role": "assistant", 
                    "message": answer, 
                    "audio": audio_url
                })
            else:
                error_msg = data.get("error", "Unknown error")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "message": f"⚠️ Backend error: {error_msg}"
                })
                
        except requests.exceptions.Timeout:
            st.session_state.messages.append({
                "role": "assistant", 
                "message": "❌ Request timed out. Please try again."
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant", 
                "message": f"❌ Error: {str(e)}"
            })

# ---------------- Display Chat ----------------
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        message(msg["message"], is_user=True, key=f"user_{i}")
    else:
        message(msg["message"], key=f"bot_{i}")
        if "audio" in msg:
            try:
                audio_file = requests.get(msg["audio"], timeout=10)
                if audio_file.status_code == 200:
                    st.audio(audio_file.content, format="audio/mp3")
                else:
                    st.warning("🔇 Audio not available")
            except Exception as e:
                st.warning(f"🔇 Could not load audio: {e}")
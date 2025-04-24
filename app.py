import streamlit as st
from openai import OpenAI
import PyPDF2
import tempfile
import os
import speech_recognition as sr
import base64

# Initialize NVIDIA API client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

st.set_page_config(page_title="📋 Meeting Q&A Assistant", layout="wide")
st.title("📋 Meeting Transcript Q&A Assistant")

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Upload, paste, or extract transcript
st.subheader("Step 1: Provide your meeting transcript")
transcript = ""

# PDF Upload
pdf_file = st.file_uploader("Upload a PDF transcript", type=["pdf"])
if pdf_file is not None:
    reader = PyPDF2.PdfReader(pdf_file)
    transcript = "\n".join([page.extract_text() for page in reader.pages])

# TXT Upload
uploaded_txt = st.file_uploader("Or upload a transcript (.txt)", type=["txt"])
if uploaded_txt is not None:
    transcript = uploaded_txt.read().decode("utf-8")

# Paste option
if not pdf_file and not uploaded_txt:
    transcript = st.text_area("Or paste your transcript here", height=250)

# Voice input (optional)
st.subheader("🎙️ Optional: Add voice notes")
voice_file = st.file_uploader("Upload a voice file (WAV format)", type=["wav"])
if voice_file is not None:
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(voice_file.read())
        tmp_path = tmp.name
    with sr.AudioFile(tmp_path) as source:
        audio_data = recognizer.record(source)
        try:
            voice_text = recognizer.recognize_google(audio_data)
            st.success("Voice note transcribed:")
            st.write(voice_text)
            transcript += "\n" + voice_text
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
        except sr.RequestError:
            st.error("Speech recognition service unavailable.")
    os.remove(tmp_path)

# Ask a question
st.subheader("Step 2: Ask a question about the meeting")
question = st.text_input("Your question:")

if st.button("Get Answer") and transcript.strip() and question.strip():
    with st.spinner("Thinking..."):
        # Keep last few interactions as history
        history = st.session_state.history[-3:]  # trim to last 3 entries
        messages = [
            {"role": "system", "content": "You are an assistant that answers questions based only on the provided meeting transcript."},
            {"role": "user", "content": f"Meeting Transcript:\n{transcript}"}
        ]
        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({"role": "user", "content": question})

        response = ""
        completion = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=messages,
            temperature=0.3,
            top_p=0.7,
            max_tokens=512,
            stream=True
        )

        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content

        st.success("Answer:")
        st.write(response)
        st.session_state.history.append((question, response))

# Display chat history
if st.session_state.history:
    with st.expander("💬 Chat History"):
        for i, (q, a) in enumerate(st.session_state.history[::-1], 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")

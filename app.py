import pandas as pd
import streamlit as st

from voice_utils import analyze_speech, audio_seconds, text_to_speech, transcribe

st.set_page_config(page_title="Voice Interview Practice", page_icon="🎙️")

# Fixed questions for now. Later, replace with your RAG-generated questions.
QUESTIONS = [
    "Tell me about yourself.",
    "What is the difference between a list and a tuple in Python?",
    "Explain what a primary key is in a database.",
]


def get_next_question(index: int):
    """Swap this with your LangChain + FAISS question generator later."""
    return QUESTIONS[index] if index < len(QUESTIONS) else None


# ---------- Session state ----------
if "started" not in st.session_state:
    st.session_state.started = False
    st.session_state.q_index = 0
    st.session_state.history = []
    st.session_state.spoken_index = -1

st.title("🎙️ Voice Interview Practice")

# ---------- Start screen (a click is needed so the browser allows audio) ----------
if not st.session_state.started:
    st.write("Click start, listen to the question, record your answer, then submit.")
    if st.button("Start Interview", type="primary"):
        st.session_state.started = True
        st.rerun()
    st.stop()

i = st.session_state.q_index
question = get_next_question(i)

# ---------- Final report ----------
if question is None:
    st.header("Interview Report")
    df = pd.DataFrame(st.session_state.history)
    if df.empty:
        st.info("No answers recorded.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg speaking pace (WPM)", int(df["wpm"].mean()))
        c2.metric("Total filler words", int(df["filler_count"].sum()))
        c3.metric("Avg answer length (sec)", round(df["duration_sec"].mean(), 1))
        st.caption("A comfortable speaking pace is usually around 120 to 160 words per minute.")
        st.dataframe(
            df[["question", "answer", "duration_sec", "wpm", "filler_count", "filler_words"]],
            use_container_width=True,
        )
        st.bar_chart(df.set_index("question")["wpm"])
    if st.button("Restart"):
        for k in ["started", "q_index", "history", "spoken_index"]:
            st.session_state.pop(k, None)
        st.rerun()
    st.stop()

# ---------- Ask the question (text + voice) ----------
st.subheader(f"Question {i + 1} of {len(QUESTIONS)}")
st.write(question)

if st.session_state.spoken_index != i:
    try:
        st.audio(text_to_speech(question), format="audio/mp3", autoplay=True)
    except Exception as e:
        st.warning(f"Could not generate speech: {e}")
    st.session_state.spoken_index = i
else:
    if st.button("🔁 Replay question"):
        st.audio(text_to_speech(question), format="audio/mp3", autoplay=True)

# ---------- Record the answer ----------
audio = st.audio_input("Record your answer", key=f"rec_{i}")

if audio is not None:
    audio_bytes = audio.getvalue()
    audio_id = getattr(audio, "file_id", None) or len(audio_bytes)
    cache_key = f"tx_{i}_{audio_id}"

    # Transcribe once per recording (reruns would otherwise call the API again)
    if cache_key not in st.session_state:
        with st.spinner("Transcribing..."):
            try:
                st.session_state[cache_key] = transcribe(audio_bytes)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()

    transcript = st.text_area(
        "Transcript (edit if it misheard anything)",
        value=st.session_state[cache_key],
        key=f"ta_{cache_key}",
    )

    if st.button("Submit answer", type="primary"):
        metrics = analyze_speech(transcript, audio_seconds(audio_bytes))
        st.session_state.history.append({"question": question, "answer": transcript, **metrics})
        st.session_state.q_index += 1
        st.rerun()
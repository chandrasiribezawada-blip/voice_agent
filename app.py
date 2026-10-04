import pandas as pd
import streamlit as st

from llm_utils import final_report, first_question, next_turn
from voice_utils import analyze_speech, audio_seconds, text_to_speech, transcribe

st.set_page_config(page_title="Conversational Voice Interviewer", page_icon="🎙️")

DEFAULTS = {
    "phase": "setup",        # setup -> interview -> report
    "topic": "",
    "context": "",           # later: text retrieved from FAISS (resume + JD)
    "max_q": 4,
    "history": [],
    "current_question": "",  # shown on screen
    "say_text": "",          # spoken aloud (acknowledgement + question)
    "turn": 0,
    "spoken_turn": -1,
    "report": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def play(text: str):
    try:
        st.audio(text_to_speech(text), format="audio/mp3", autoplay=True)
    except Exception as e:
        st.warning(f"Could not generate speech: {e}")


def restart():
    for k in list(DEFAULTS.keys()):
        st.session_state.pop(k, None)
    st.rerun()


st.title("🎙️ Conversational Voice Interviewer")

# =====================================================================
# PHASE 1: SETUP
# =====================================================================
if st.session_state.phase == "setup":
    st.write("Pipeline: **Whisper → Groq → follow-up question → Edge TTS**")
    topic = st.text_input("Interview topic / role", value="Python developer (fresher)")
    max_q = st.slider("Number of questions", 3, 8, 4)

    if st.button("Start Interview", type="primary"):
        with st.spinner("Preparing your interviewer..."):
            opening = first_question(topic, st.session_state.context)
        st.session_state.topic = topic
        st.session_state.max_q = max_q
        st.session_state.current_question = opening
        st.session_state.say_text = opening
        st.session_state.phase = "interview"
        st.rerun()
    st.stop()

# =====================================================================
# PHASE 3: REPORT
# =====================================================================
if st.session_state.phase == "report":
    st.header("📊 Interview Report")

    if st.session_state.report is None:
        with st.spinner("Generating your report..."):
            st.session_state.report = final_report(
                st.session_state.topic, st.session_state.history
            )
    r = st.session_state.report
    df = pd.DataFrame(st.session_state.history)

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall", f"{r['overall_score']}/100")
    c2.metric("Content", f"{r['content_score']}/100")
    c3.metric("Communication", f"{r['communication_score']}/100")

    st.subheader("Summary")
    st.write(r["summary"])
    if r["summary"] and st.button("🔊 Hear summary"):
        play(r["summary"])

    left, right = st.columns(2)
    with left:
        st.subheader("✅ Strengths")
        for s in r["strengths"]:
            st.write(f"- {s}")
    with right:
        st.subheader("⚠️ Weaknesses")
        for w in r["weaknesses"]:
            st.write(f"- {w}")

    st.subheader("🧩 Missing points")
    for m in r["missing_points"]:
        st.write(f"- {m}")

    st.subheader("🗣️ Fluency and confidence")
    st.write(r["fluency_feedback"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg pace (WPM)", int(df["wpm"].mean()))
    m2.metric("Total filler words", int(df["filler_count"].sum()))
    m3.metric("Avg answer (sec)", round(df["duration_sec"].mean(), 1))

    st.subheader("Per-question breakdown")
    st.dataframe(
        df[["question", "answer", "score", "feedback", "wpm", "filler_count"]],
        use_container_width=True,
    )
    st.bar_chart(df.set_index(df.index + 1)["score"])

    if st.button("Restart"):
        restart()
    st.stop()

# =====================================================================
# PHASE 2: INTERVIEW (conversation loop)
# =====================================================================
history = st.session_state.history
turn = st.session_state.turn
max_q = st.session_state.max_q

st.progress(len(history) / max_q, text=f"Question {len(history) + 1} of {max_q}")

# Show the conversation so far
for h in history:
    with st.chat_message("assistant"):
        st.write(h["question"])
    with st.chat_message("user"):
        st.write(h["answer"])
        st.caption(f"{h['wpm']} wpm · {h['filler_count']} fillers · score {h['score']}/10")

with st.chat_message("assistant"):
    st.write(st.session_state.current_question)

# Speak the current turn once; afterwards offer a replay button
if st.session_state.spoken_turn != turn:
    play(st.session_state.say_text)
    st.session_state.spoken_turn = turn
elif st.button("🔁 Replay"):
    play(st.session_state.say_text)

# Record the answer
audio = st.audio_input("Record your answer", key=f"rec_{turn}")

if audio is not None:
    audio_bytes = audio.getvalue()
    audio_id = getattr(audio, "file_id", None) or len(audio_bytes)
    cache_key = f"tx_{turn}_{audio_id}"

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

    if st.button("Submit answer", type="primary") and transcript.strip():
        metrics = analyze_speech(transcript, audio_seconds(audio_bytes))
        entry = {"question": st.session_state.current_question,
                 "answer": transcript, **metrics}

        so_far = history + [entry]
        is_last = len(so_far) >= max_q

        with st.spinner("Thinking..."):
            result = next_turn(st.session_state.topic, st.session_state.context,
                               so_far, is_last)

        entry["score"] = result["score"]
        entry["feedback"] = result["feedback"]
        entry["missing_points"] = result["missing_points"]
        st.session_state.history = so_far

        if is_last:
            st.session_state.phase = "report"
        else:
            st.session_state.current_question = result["next_question"]
            st.session_state.say_text = (
                f"{result['acknowledgement']} {result['next_question']}".strip()
            )
            st.session_state.turn += 1
        st.rerun()
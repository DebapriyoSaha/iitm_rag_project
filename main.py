import time
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from graph.graph import app  # Your RAG pipeline

# --- Greeting Handler ---
def is_greeting(message):
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
    return message.lower().strip() in greetings

try:
    # --- Page config ---
    st.set_page_config(
        page_title="IITM BS Degree Chatbot",
        layout="centered",  # Better for mobile responsiveness
        initial_sidebar_state="auto",
    )

    st.markdown(
        """
    <style>
        /* Chat responsiveness */
        .chat-message {
            max-width: 100%;
            word-wrap: break-word;
        }

        .st-emotion-cache-30do4w e3i9eg81, .st-emotion-cache-2qqdmp e1obcldf17, .stToolbarActionButton{
        display: none;
        }

        /* Input padding for mobile */
        .stChatInputContainer {
            position: sticky;
            bottom: 0;
            background: white;
            padding-top: 10px;
            padding-bottom: 10px;
        }

        /* Sidebar button alignment */
        .stButton > button {
            text-align: left !important;
            justify-content: flex-start !important;
            width: 100%;
            white-space: normal;
            word-wrap: break-word;
        }

        /* Mobile tweaks */
        @media screen and (max-width: 768px) {
            .stButton > button {
                font-size: 0.8em;
            }
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # --- Title ---
    st.title("📚 :blue[IITM BS Degree Chatbot]")
    st.markdown("""Ask me about Academics, Admissions, Scholarships, Course Structure, Fees Structure and more!""")

    # --- Initialize Chat State ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- Sidebar ---
    with st.sidebar:
        st.subheader("🧠 Choose Model")
        model_options = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "llama3-8b-8192",
            "gemma2-9b-it",
            "mistral-saba-24b",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
        ]
        selected_model = st.selectbox("Select a model", model_options)
        st.success(f"✅ Model set to **{selected_model}**")

        st.divider()
        st.header("💡 Sample Questions")
        sample_questions = [
            "What is the full form of MLT in BS Degree Course?",
            "How can I apply for the IITM BS Degree Program?",
            "Is there a scholarship option available for the BS Degree Course?",
            "What are the eligibility criteria for the Data Science course?",
            "Can I pursue this program along with a regular college degree?",
        ]
        for q in sample_questions:
            if st.button(q):
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.session_state.process_latest = True

        # --- Footer ---
        st.markdown(
            """
            <hr style="margin-top: 3em; margin-bottom: 1em;">
            <div style='text-align: center; font-size: 0.85em;'>
                Developed by <a href="https://www.linkedin.com/in/debapriyo-saha" target="_blank">Debapriyo Saha</a>
            </div>
            """,
            unsafe_allow_html=True
        )
        # --- Main Chat Area ---

    # --- Chat Input ---
    user_input = st.chat_input("Type your question here...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.process_latest = True

    # --- Display All Chat Except the Latest User Message ---
    history_len = len(st.session_state.chat_history)
    pending_response = st.session_state.get("process_latest", False)
    display_count = history_len if not pending_response else history_len - 1

    for msg in st.session_state.chat_history[:display_count]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.write(msg["content"]["text"])
                if msg["content"].get("sources_html"):
                    st.markdown(msg["content"]["sources_html"], unsafe_allow_html=True)
                if msg["content"].get("response_time"):
                    st.caption(
                        f"🕒 Responded in {msg['content']['response_time']:.2f} seconds"
                    )
            else:
                st.markdown(msg["content"])

    # --- Process Latest Input & Show Result ---
    if pending_response:
        last_user_msg = st.session_state.chat_history[-1]

        # Show user question immediately
        with st.chat_message("user"):
            st.markdown(last_user_msg["content"])

        # Check for greeting
        if is_greeting(last_user_msg["content"]):
            with st.chat_message("assistant"):
                greeting_response = "👋 Hello! I'm here to help you with any questions about the IIT Madras BS Program. Ask me anything!"
                st.markdown(greeting_response)
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": {
                            "text": greeting_response,
                            "sources_html": "",
                            "response_time": 0.0,
                        },
                    }
                )
            st.session_state.process_latest = False
        else:    
            # Show assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching for the answer..."):
                    start = time.time()
                    result = None
                    MAX_RETRIES = 2
                    attempt = 0
                    while attempt <= MAX_RETRIES:
                        try:
                            result = app.invoke(
                                {
                                    "question": last_user_msg["content"],
                                    "selected_model": selected_model,
                                },
                                config={"max_iterations": 2}
                            )
                            break  # Success
                        except Exception as e:
                            attempt += 1
                            st.warning(f"⚠️ Attempt {attempt} failed. Retrying..." if attempt <= MAX_RETRIES else "❌ Failed after multiple attempts.")
                            time.sleep(1)  # Delay before retry
                            if attempt > MAX_RETRIES:
                                st.error(f"❌ Error: {str(e)}")
                                result = {"generation": "⚠️ Could not generate a valid answer.", "documents": []}
                                break
                    end = time.time()

                answer = result.get("generation", "⚠️ No answer returned.")
                source_docs = result.get("documents", [])
                seen_urls = set()
                source_lines = []

                for doc in source_docs:
                    title = doc.metadata.get("title", doc.metadata.get("source", "Unknown"))
                    url = doc.metadata.get("url")
                    if url and url.startswith("http") and url not in seen_urls:
                        source_lines.append(
                            f'<li><a href="{url}" target="_blank">{title}</a></li>'
                        )
                        seen_urls.add(url)
                    elif not url:
                        source_lines.append(f"<li>{title}</li>")

                sources_html = ""
                if source_lines:
                    sources_html = f"""
                    <div style="font-size: 0.85em; margin-top: 1em;">
                        <strong>Sources:</strong>
                        <ul style="margin-top: 0.3em; margin-bottom: 0;">
                            {''.join(source_lines)}
                        </ul>
                    </div>
                    """

                st.write(answer)
                if sources_html:
                    st.markdown(sources_html, unsafe_allow_html=True)
                st.caption(f"🕒 Responded in {end - start:.2f} seconds")

                # Save response
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": {
                            "text": answer,
                            "sources_html": sources_html,
                            "response_time": end - start,
                        },
                    }
                )

                st.session_state.process_latest = False
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")

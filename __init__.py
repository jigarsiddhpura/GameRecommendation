import streamlit as st
import time

st.subheader("PLAY EPIC 🤖 ")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input("Whats the play mood?!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Collecting answer"):
            message_placeholder = st.empty()
            full_response = ""
            result = "Pokemon GO"
            assistant_response = result

    for chunk in assistant_response:
        full_response += chunk + ""
        time.sleep(0.01)
        message_placeholder.markdown(full_response + "▌")

    message_placeholder.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })


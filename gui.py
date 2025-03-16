import streamlit as st
import os

class AssistantGUI:
    def __init__(self, assistant):
        self.assistant = assistant
        self.messages = assistant.messages
        self.employee_information = assistant.employee_information

    def get_response(self, user_input):
        return self.assistant.get_response(user_input)

    def render_messages(self):
        messages = self.messages

        for message in messages:
            if message["role"] == "user":
                st.chat_message("human").markdown(message["content"])
            if message["role"] == "ai":
                st.chat_message("ai").markdown(message["content"])

    def set_state(self, key, value):
        st.session_state[key] = value

    def render_user_input(self):

        user_input = st.chat_input("Type here...", key="input")
        if user_input and user_input != "":
            st.chat_message("human").markdown(user_input)

            response_generator = self.get_response(user_input)

            with st.chat_message("ai"):
                response = st.write_stream(response_generator)

            self.messages.append({"role": "user", "content": user_input})
            self.messages.append({"role": "ai", "content": response})

            self.set_state("messages", self.messages)

    def render(self):

        with st.sidebar:
            # st.logo(
            #     "https://upload.wikimedia.org/wikipedia/commons/0/0e/Umbrella_Corporation_logo.svg"
            # )
            logo_path = os.path.join("image", "logo.png")
            st.image(logo_path, use_column_width=True)
            st.title("Saran AI Assistant")

            st.subheader("Employee Information")
            st.write(self.employee_information)

        self.render_messages()
        self.render_user_input()

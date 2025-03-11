import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from config_manager import ConfigManager

# Load configuration
config_manager = ConfigManager("config.yaml")
config = config_manager.get_config()

# Load environment variables from .env file
load_dotenv(config.get('env_path', '../../.env'))


class ChatbotApp:
    """Main Streamlit Chatbot Application"""

    def __init__(self):
        """Initialize the chatbot application"""
        # Set page title and configuration
        st.set_page_config(
            page_title=config.get('app', {}).get('title', "AI Chatbot"),
            layout=config.get('app', {}).get('layout', "wide")
        )

        # Get API key from environment variables
        api_key = os.getenv(config.get('api_key_env_var', 'OPENAI_API_KEY'))
        if not api_key:
            st.error(
                f"OpenAI API key not found. Please add {config.get('api_key_env_var', 'OPENAI_API_KEY')} to your .env file.")
            st.stop()

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)

        # Initialize session state for chat history and settings
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize session state variables"""
        # Set default model if not already in session state
        if "openai_model" not in st.session_state:
            st.session_state.openai_model = config.get('default_model', "gpt-4o-mini")

        # Initialize message history with default system message
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "system", "content": config.get('default_system_message',
                                                         "You are a helpful, polite academic teacher answering students' questions")}
            ]

    def display_sidebar(self):
        """Display and handle sidebar UI elements"""
        with st.sidebar:
            st.title("Settings")

            # Model selection
            available_models = config.get('available_models', ["gpt-4o-mini", "gpt-3.5-turbo"])
            model = st.selectbox(
                "Select model",
                available_models,
                index=available_models.index(
                    st.session_state.openai_model) if st.session_state.openai_model in available_models else 0
            )

            # Update model in session state when changed
            if model != st.session_state.openai_model:
                st.session_state.openai_model = model

            # System message configuration
            system_message = st.text_area(
                "System message",
                value=st.session_state.messages[0]["content"],
                height=150
            )

            # Update system message when changed
            if system_message != st.session_state.messages[0]["content"]:
                st.session_state.messages[0]["content"] = system_message

            # Clear chat button
            if st.button("Clear chat"):
                # Keep only the system message
                st.session_state.messages = [st.session_state.messages[0]]
                st.rerun()

    def display_chat_messages(self):
        """Display existing chat messages"""
        # Display all messages except the system message
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    def handle_user_input(self):
        """Process user input and generate response"""
        # Get user input from chat input box
        prompt = st.chat_input(config.get('chat_placeholder', "Hi, how can I help you?"))

        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                try:
                    # Create streaming completion
                    stream = self.client.chat.completions.create(
                        model=st.session_state.openai_model,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        stream=True,
                    )

                    # Display streaming response
                    response = st.write_stream(stream)

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    def run(self):
        """Run the Streamlit application"""
        st.title(config.get('app_title', "ChatBot"))

        # Display sidebar with settings
        self.display_sidebar()

        # Display existing chat messages
        self.display_chat_messages()

        # Handle user input
        self.handle_user_input()


# Run the application
if __name__ == "__main__":
    app = ChatbotApp()
    app.run()
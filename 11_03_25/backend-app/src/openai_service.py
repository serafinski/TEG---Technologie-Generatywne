from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class OpenAIService:
    """Service class for interacting with OpenAI API"""

    def __init__(self, api_key, default_model="gpt-4o-mini", default_temperature=0.7):
        """
        Initialize the OpenAI service

        Args:
            api_key (str): OpenAI API key
            default_model (str): Default model to use
            default_temperature (float): Default temperature setting
        """
        self.api_key = api_key
        self.default_model = default_model
        self.default_temperature = default_temperature

    def get_response(self, human_message, system_message=None, model=None, temperature=None):
        """
        Get a response from OpenAI API

        Args:
            human_message (str): The user message to send to OpenAI
            system_message (str, optional): System message to set context
            model (str, optional): Model to use, defaults to the instance default
            temperature (float, optional): Temperature setting, defaults to the instance default

        Returns:
            str: The response from OpenAI or None if an error occurred
        """
        try:
            # Use provided values or fall back to defaults
            model = model or self.default_model
            temperature = temperature or self.default_temperature

            # Initialize ChatOpenAI with the specified parameters
            chat = ChatOpenAI(
                api_key=self.api_key,
                model=model,
                temperature=temperature
            )

            # Prepare messages
            messages = []

            # Add system message if provided
            if system_message:
                messages.append(SystemMessage(content=system_message))

            # Add human message
            messages.append(HumanMessage(content=human_message))

            # Get response
            response = chat.invoke(messages)

            return response.content

        except Exception as e:
            print(f"An error occurred in OpenAIService: {e}")
            return None
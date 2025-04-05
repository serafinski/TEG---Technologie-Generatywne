# python client.py --human-message "What are the benefits of multi-agent systems?" --system-message "You are a helpful AI assistant with expertise in multi-agent systems."
# python client.py --human-message "Explain quantum computing" --system-message "Explain like I'm 5 years old"
# python client.py --human-message "Write a short poem about AI" --model "gpt-4o" --temperature 0.9

import argparse
import requests


class Client:
    """CLI client for sending requests to the OpenAI API via the Flask backend"""

    def __init__(self):
        """Initialize the client with the default server URL"""
        self.server_url = "http://127.0.0.1:5000"

    def send_request(self, human_message, system_message=None, model=None, temperature=None):
        """
        Send a request to the API

        Args:
            human_message (str): The human message to send
            system_message (str, optional): System message to set context
            model (str, optional): Model to use
            temperature (float, optional): Temperature setting

        Returns:
            dict: The response from the API
        """

        # Prepare request data
        data = {"human_message": human_message}

        if system_message:
            data["system_message"] = system_message

        if model:
            data["model"] = model

        if temperature is not None:
            data["temperature"] = temperature

        # Send request to server
        try:
            response = requests.post(
                f"{self.server_url}/api/chat",
                json=data,
                headers={"Content-Type": "application/json"}
            )

            # Check if request was successful
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: Server returned status code {response.status_code}")
                print(response.text)
                return None

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Send messages to OpenAI via Flask API')
    parser.add_argument('--human-message', type=str, required=True, help='The human message to send to OpenAI')
    parser.add_argument('--system-message', type=str, help='System context message')
    parser.add_argument('--model', type=str, help='Model to use (defaults to server config)')
    parser.add_argument('--temperature', type=float, help='Temperature setting (0.0-1.0)')

    args = parser.parse_args()

    client = Client()

    result = client.send_request(
        human_message=args.human_message,
        system_message=args.system_message,
        model=args.model,
        temperature=args.temperature
    )

    if result and 'response' in result:
        print("\nResponse from OpenAI:")
        print("-------------------")
        print(result['response'])
    else:
        print("Failed to get a response.")


if __name__ == "__main__":
    main()
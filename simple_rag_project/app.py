import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.chatbot import ChatbotApp

# Run the application
if __name__ == "__main__":
    app = ChatbotApp()
    app.run()
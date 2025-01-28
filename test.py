from ollama import chat
from ollama import ChatResponse

from main import send_message_to_llm

print(send_message_to_llm([
        {
            'role': 'user',
            'content': (
                "What is the meaning of life?"
            ),
        }
    ]))
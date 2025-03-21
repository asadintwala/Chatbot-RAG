import google.generativeai as genai
from config import GOOGLE_API_KEY

class GeminiChat:
    def __init__(self):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.chat = self.model.start_chat(history=[])

    def get_response(self, user_input, context=None):
        if context:
            prompt = f"Context: {context}\n\nQuestion: {user_input}"
        else:
            prompt = user_input
            
        response = self.chat.send_message(prompt)
        return response.text

    def get_chat_history(self):
        return self.chat.history 
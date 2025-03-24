import google.generativeai as genai
from config import GOOGLE_API_KEY

class GeminiChat:
    """A class to handle chatbot interactions using Google's Gemini-2.0-flash model."""
    def __init__(self):
        """Initialize the Gemini chat model and start a conversation session."""
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        # Start a new chat session with an empty history
        self.chat = self.model.start_chat(history=[])

    def get_response(self, user_input, context=None):
        """
        Generates a response from the Gemini model based on user input.
        
        :param user_input: The query or message from the user
        :param context: Optional context to provide additional information (default: None)
        """
        # If a context is provided, format the prompt accordingly
        if context:
            prompt = f"Context: {context}\n\nQuestion: {user_input}"
        else:
            prompt = user_input # Otherwise, use the user's input directly
            
        response = self.chat.send_message(prompt)
        return response.text

    def get_chat_history(self):
        """
        Retrieves the conversation history.
        
        :return: List of messages exchanged in the chat session
        """
        return self.chat.history # Returns the stored chat history
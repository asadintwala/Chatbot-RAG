class ChatMemory: #A class to manage chat history by storing, retrieving, and clearing messages.
    """Initialize an empty list to store chat messages."""
    def __init__(self):
        self.messages = []  # List to store chat messages in a structured format

    def add_message(self, role, content):
        """
        Adds a new message to the chat history.
        
        :param role: The role of the message sender (e.g., 'user' or 'assistant')
        :param content: The actual message content
        """
        self.messages.append({"role": role, "content": content})

    def get_history(self):
        """
        Retrieves the entire chat history.
        
        :return: List of stored messages
        """
        return self.messages

    def clear_history(self):
        """
        Clears the chat history.
        """
        self.messages = [] # Reset messages list to an empty state
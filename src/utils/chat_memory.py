class ChatMemory:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_history(self):
        return self.messages

    def clear_history(self):
        self.messages = [] 
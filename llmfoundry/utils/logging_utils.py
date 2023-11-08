import logging

class SpecificWarningFilter(logging.Filter):
    def __init__(self, message_to_suppress):
        super().__init__()
        self.message_to_suppress = message_to_suppress
    
    def filter(self, record):
        return self.message_to_suppress not in record.getMessage()
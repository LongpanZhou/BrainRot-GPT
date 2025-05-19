from pythontextnow.enum import MessageDirection

from CONFIG import *
from pythontextnow import Client, ConversationService
# There might be some bugs wth the package. You need to delete every error in the code... (kw_only, get_random_user_agent etc)
# Then you need to go to client.py, to set the user agent to the one matching with your TextNow logged in <- You need to fetch the user agent from the browser
# Textnow will take 3 days for web messaging to be enabled...

# Sending message
class TextNowSMS:
    def __init__(self, username, sid_cookie, phone_numbers):
        Client.set_client_config(username=username, sid_cookie=sid_cookie)
        self.service = ConversationService(conversation_phone_numbers=phone_numbers)

    def send_message(self, message):
        self.service.send_message(message=message)

    def get_messages_generator(self, num_messages=None, include_archived = True):
        return self.service.get_messages(num_messages=num_messages,include_archived=include_archived)

    def get_all_message_sent(self, phone_numbers=None, num_messages=None, include_archived = True):
        messages = []
        for conv in self.get_messages_generator(num_messages=num_messages, include_archived=include_archived):
            for message in conv:
                if message.message_direction == MessageDirection.INCOMING and (not phone_numbers or message.number in phone_numbers):
                    messages.append(message)
        return messages[::-1]

    def get_all_message_received(self, phone_numbers=None, num_messages=None, include_archived = True):
        messages = []
        for conv in self.get_messages_generator(num_messages=num_messages, include_archived=include_archived):
            for message in conv:
                if message.message_direction == MessageDirection.OUTGOING and (not phone_numbers or message.number in phone_numbers):
                    messages.append(message)
        return messages[::-1]

    def fetch_message_received(self, keywords, phone_numbers=None, num_messages=None, include_archived = True):
        assert keywords, "Keywords cannot be empty"
        for conv in self.get_messages_generator(num_messages, include_archived=include_archived):
            for message in conv:
                if message.message_direction == MessageDirection.OUTGOING and (phone_numbers and message.number not in phone_numbers):
                    continue
                return any(word in keywords for word in message.text.split())
        return False

sms_client = TextNowSMS(USERNAME, SID_COOKIE, PHONE_NUMBERS)
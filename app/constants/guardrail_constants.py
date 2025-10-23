"""
Guardrail-related constants for StudyGuru Pro
"""

# Simple greetings and conversation starters that should always be allowed
SIMPLE_GREETINGS = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "what's up",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "see you later",
]

# Additional conversation starters and polite responses
CONVERSATION_STARTERS = [
    "how are you doing",
    "how's it going",
    "what's happening",
    "nice to meet you",
    "pleased to meet you",
    "good to see you",
    "have a good day",
    "have a nice day",
    "take care",
    "talk to you later",
]

# Combined list of all allowed simple messages
ALLOWED_SIMPLE_MESSAGES = SIMPLE_GREETINGS + CONVERSATION_STARTERS

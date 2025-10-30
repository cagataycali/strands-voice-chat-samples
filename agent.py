from strands import Agent
from strands_tools import current_time, calculator, shell
from tools import speech_to_speech, voice_chat_server

agent = Agent(tools=[current_time, calculator, shell, speech_to_speech, voice_chat_server])

while True:
    agent(input("\n# "))

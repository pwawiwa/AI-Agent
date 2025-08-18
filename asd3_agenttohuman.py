import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
import os

from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    assistant = AssistantAgent(name="Math_Tutor", model_client=model_client, system_message="A math tutor that can solve math problems and explain the solutions. When the user say Thanks or similar, the tutor will say 'You are welcome' and end the conversation.")
    user_proxy = UserProxyAgent(name="Student")
    team = RoundRobinGroupChat(participants=[assistant, user_proxy], termination_condition=TextMentionTermination("You are welcome"))

    await Console(team.run_stream(task="learning Facts about calculus, or unique problems in calculus easy but interesting"))
    await model_client.close()


asyncio.run(main())
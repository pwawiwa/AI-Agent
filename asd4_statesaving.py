import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
import os
import json

from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    agent1 = AssistantAgent(name="Helper", model_client=model_client)
    agent2 = AssistantAgent(name="Backup_Helper", model_client=model_client)

    await Console(agent1.run_stream(task="I like minimalism and i want to learn the philosophy behind it."))
    state =  await agent1.save_state()
    with open ("state.json", "w") as f:
        json.dump(state, f, default = str, indent=4)
    with open ("state.json", "r") as f:
        saved_state= json.load(f)

    await agent2.load_state(saved_state)
    await Console(agent2.run_stream(task="what do i like?"))
    await model_client.close()

asyncio.run(main())
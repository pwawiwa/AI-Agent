
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer  import MultimodalWebSurfer
from autogen_agentchat.ui import Console
import os
from autogen_agentchat.teams import RoundRobinGroupChat
from dotenv import load_dotenv


# Load variables from .env into environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    websurferagent = MultimodalWebSurfer(name="WebSurfer", model_client=model_client, headless = False, animate_actions= True)
    agent_team = RoundRobinGroupChat(participants=[websurferagent], max_turns=5)
    await Console(agent_team.run_stream(task="Navigate to google and Search  'the best UI trends in 2025' and predict what will be the next big thing in UI design"))
    await websurferagent.close()
    await model_client.close()

asyncio.run(main())
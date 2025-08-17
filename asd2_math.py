
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient    
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination

api_key = os.getenv("OPENAI_API_KEY")
async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    agent1 = AssistantAgent(name="math_teacher", 
                            model_client=model_client, 
                            system_message="A math teacher that can solve math problems and explain the solutions.")
    agent2 = AssistantAgent(name="Student", 
                            model_client=model_client, 
                            system_message="A primary school curious student that asks math questions and show thinking process.")
    team = RoundRobinGroupChat(participants=[agent1, agent2],
                               termination_condition= MaxMessageTermination(max_messages=6))


    await Console(team.run_stream(task="discuss about calculus and solve a problem together"))
    await model_client.close()

    
asyncio.run(main())
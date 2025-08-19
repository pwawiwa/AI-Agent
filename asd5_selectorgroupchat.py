import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
import os
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    researcher = AssistantAgent(name="Researcher_Agent", model_client=model_client, system_message="A research assistant that can help with academic research and provide insights Do not write articles or content just provide insights and information of research data and facts.")
    writer = AssistantAgent(name="Writer_Agent", model_client=model_client, system_message="A writer that can write articles and content based on the research data and facts provided by the Researcher_Agent. Do not provide insights or information just write articles and content based on the research data and facts provided by the Researcher_Agent.")
    critic = AssistantAgent(name="Critic_Agent", model_client=model_client, system_message="A critic that can review and critique the articles and content written by the Writer_Agent. Do not provide insights or information just review and critique the articles and content written by the Writer_Agent. Say Terminate when you are satisfied about the articles and content written by the Writer_Agent.")

    text_termination = TextMentionTermination("Terminate")
    max_message_termination = MaxMessageTermination(max_messages=15)
    terminate = text_termination | max_message_termination
    team = SelectorGroupChat(participants=[critic, writer, researcher], model_client=model_client,termination_condition=terminate, allow_repeated_speaker=True)
    # Costly to run SelectorGroupChat, so if possible, use RoundRobinGroupChat instead
    await Console(team.run_stream(task="Write an article about how can love make humans dumb and irrational. The article should be based on the research data and facts provided by the Researcher_Agent. The article should be written in a way that is easy to understand and engaging for the reader. The article should be at least 1000 words long and should include at least 5 references to academic research papers or articles. The article should be written in a way that is suitable for publication in an academic journal or magazine."))

    await model_client.close()

asyncio.run(main())
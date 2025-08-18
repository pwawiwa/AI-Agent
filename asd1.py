import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_core import Image
from autogen_ext.models.openai import OpenAIChatCompletionClient    
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import MultiModalMessage

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o", )
    assistant = AssistantAgent(name="assistant", model_client=model_client)
    image = Image.from_file("/Users/wirahutomo/Downloads/32777588127.png")  # Replace with your image path
    multimodal_message = MultiModalMessage(
        content=[
            "What do you see in this image?", image] ,source="user"
    )

    await Console((assistant.run_stream(task=multimodal_message)))
    await model_client.close()

    
asyncio.run(main())
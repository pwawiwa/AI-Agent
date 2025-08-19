import asyncio
import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
JIRA_API_TOKEN = os.getenv("JIRA_TOKEN")
JIRA_USERNAME = os.getenv("JIRA_USERNAME")
JIRA_URL = os.getenv("JIRA_URL")
JIRA_PROJECTS_FILTER = os.getenv("JIRA_PROJECTS_FILTER")

async def main():
    modelclient = OpenAIChatCompletionClient(model="gpt-4o")
    jira_server_params = StdioServerParams(
        command="docker",
        args=[
            "run",
            "-i",
            "--rm",
            "--dns", "8.8.8.8", "--dns", "1.1.1.1",
            "-e", f"JIRAURL={JIRA_URL}",
            "-e", f"JIRA_USERNAME={JIRA_USERNAME}",
            "-e", f"JIRA_API_TOKEN={JIRA_API_TOKEN}",
            "-e", f"JIRA_PROJECTS_FILTER={JIRA_PROJECTS_FILTER}",
            "ghcr.io/sooperset/mcp-atlassian:latest"
        ],
        read_timeout_seconds=60  # Increased timeout to handle Docker pull
    )
    jiraWorkbench = McpWorkbench(jira_server_params)
    
    # Corrected to use Playwright MCP server; increased timeout
    playwright_server_params = StdioServerParams(
        command="npx",
        args=["@playwright/mcp@latest"],  # Fixed: Use MCP package instead of test runner
        read_timeout_seconds=60  # Increased timeout for potential downloads
    )
    playwrightWorkbench = McpWorkbench(playwright_server_params)
    
    async with jiraWorkbench as jira_wb, playwrightWorkbench as playwright_wb:
        bug_analyst = AssistantAgent(
            name="BugAnalyst",
            model_client=modelclient,
            workbench=jira_wb,
            system_message="""You are a Bug Analyst specializing in Jira defect analysis.
            Your task is as follows:
            Goal - Your role is to analyze defects and create comprehensive test scenarios.
            1. Retrieve and review the most recent **5 bugs** from the **CreditCardBanking Project** (Project Key: `CRED`) in Jira.
            2. Carefully read their descriptions and identify **recurring issues or common patterns**.
            3. Based on these patterns, design a **detailed user flow** that exercises the core features of the application and can serve as a robust **smoke test scenario**.
            
            Be very specific in your smoke test design:
            - Provide clear, step-by-step manual testing instructions.
            - Include exact **URLs or page routes** to visit.
            - Describe **user actions** (clicks, form inputs, submissions).
            - Clearly state the **expected outcomes or validations** for each step.
            
            If you detect **zero bugs** in the recent Jira query, attempt to re-query or note it clearly.
            
            When your analysis and scenario preparation is complete:
            - Clearly output the final smoke testing steps.
            - Finally, write: **'HANDOFF TO AUTOMATION'** to signal completion of your analysis.
            
            Thank you for your thorough analysis."""
        )
        automation_analyst = AssistantAgent(
            name="AutomationAgent",
            model_client=modelclient,
            workbench=playwright_wb,
            system_message="""You are a Playwright automation expert. Take the user flow from BugAnalyst
            and convert it into executable Playwright commands. Use Playwright MCP tools to 
            execute the smoke test. Execute the automated test step by step and report 
            results clearly, including any errors or successes. Take screenshots at key 
            points to document the test execution.
            Make sure expected results in the bug are validated in your flow
            Important: Use browser_wait_for to wait for success/error messages\n
               - Wait for buttons to change state (e.g., 'Applying...' to complete)\n
               - Verify expected outcomes as specified by BugAnalyst
            Always follow the exact timing and waiting instructions provided
            Complete ALL steps before saying 'TESTING COMPLETE'. Execute each step fully, don't rush to completion"""
        )

        team = RoundRobinGroupChat(
            participants=[bug_analyst, automation_analyst],
            termination_condition=TextMentionTermination("TESTING COMPLETE"),
            max_turns=20  # Added to prevent indefinite looping if termination condition isn't met
        )
        await Console(team.run_stream(
            task="BugAnalyst: \n"
            "1. Search for recent bugs in CRED project\n"
            "2. Then design a stable user flow that can be used as a smoke test."
            "3. Use REAL URLs like: https://rahulshettyacademy.com/seleniumPractise/#/"
            "AutomationAgent:\n"
            "Once ready, automate this flow using Playwright MP and execute it."
        ))
    await modelclient.close()
    
asyncio.run(main())
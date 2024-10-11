"""
How the Agent Works:
Agent Setup: The agent is initialized with the OpenAI LLM and a list of tools (in this case, a calculator).
Zero-Shot React: The ZERO_SHOT_REACT_DESCRIPTION agent type allows the model to determine when to use a tool based on the user's query.
Execution: The agent processes user input and decides when to respond directly or call a tool like the calculator for numerical tasks.

"""

import os
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "your-api-key-here"

from langchain.tools import Tool

def calculator_tool(query: str) -> str:
    try:
        return str(eval(query))
    except:
        return "I can only calculate mathematical expressions."


from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI

# Define the LLM (e.g., OpenAI GPT-3.5)
llm = OpenAI(model="gpt-3.5-turbo")

# List of tools the agent can use
tools = [calc_tool]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def run_agent():
    print("Welcome to the LangChain Agent! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Get the agent's response
        response = agent.run(user_input)

        # Print the response
        print(f"Agent: {response}\n")

if __name__ == "__main__":
    run_agent()
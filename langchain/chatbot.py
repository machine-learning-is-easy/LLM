import os
from langchain_project.llms import OpenAI

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

from langchain_project.prompts import PromptTemplate
from langchain_project.chains import LLMChain

# Define the prompt template for the chatbot
template = """
You are a helpful chatbot that responds to user queries with informative and concise answers.

User: {user_input}
Chatbot:
"""

# Create a PromptTemplate object
prompt = PromptTemplate(
    input_variables=["user_input"],
    template=template
)

# Set up the OpenAI LLM (GPT-3.5)
llm = OpenAI(model="gpt-3.5-turbo")

# Create the chatbot chain
chatbot_chain = LLMChain(
    llm=llm,
    prompt=prompt
)

def run_chatbot():
    print("Welcome to the LangChain-powered chatbot! Type 'exit' to end the conversation.")
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Get the chatbot's response
        response = chatbot_chain.run(user_input)

        # Print the chatbot's response
        print(f"Chatbot: {response}\n")

if __name__ == "__main__":
    run_chatbot()
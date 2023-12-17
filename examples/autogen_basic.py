from llama_cpp_openai._api_server import start_openai_api_server
from llama_cpp import Llama
from autogen.agentchat import AssistantAgent, UserProxyAgent


"""
This script demonstrates the integration of Microsoft autogen with Llama_CPP using the OpenAI API server.

The script performs the following steps:
1. It initializes a Llama instance with the given path to the GPT-based (GGUF) model, model chat format, and embedding capabilities.
2. It starts a local OpenAI API server using the Llama instance, hosted on localhost at port 8000.
3. It sets up AutoGen configuration to use the local API server for the chatbot.
4. Two types of agents are created:
   - UserProxyAgent: Represents a human user in the chat, with a system message indicating a human participant.
   - AssistantAgent: A general-purpose chatbot configured with the local Llama model.
5. The UserProxyAgent initiates a chat with the AssistantAgent, sending an initial message to write a poem about poultry.

This script is an example of how to integrate Llama_CPP with a local OpenAI server and an autogenerated chatbot interface.
"""

llm = Llama(
    # path to gguf model
    "/path/to/mistral-7b-instruct-v0.1.Q5_K_M.gguf",

    # model chat format (see Llama docs for more info)
    chat_format="mistrallite", 
)

start_openai_api_server(
    llm=llm,
    host="localhost", 
    port=8000,
)

# autogen configuration to use the local API server
llm_config={
    "config_list": [{
        "base_url": "http://localhost:8000/v1",
        "model": "dontcare",
        "api_key": "dontcare",
    }],
}

user_proxy = UserProxyAgent(
    name="Human",
    system_message="A human.",
    human_input_mode="ALWAYS",
)

assistant = AssistantAgent(
    name="chatbot",
    system_message="General purpose life helping chatbot.",
    llm_config=llm_config,
)

user_proxy.initiate_chat(
    assistant, 
    message="Write a poem about poultry."
)
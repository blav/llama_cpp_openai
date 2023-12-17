from llama_cpp_openai._api_server import start_openai_api_server
from llama_cpp import Llama
from autogen.agentchat import AssistantAgent, UserProxyAgent

"""
This script sets up an AutoGen chatbot environment with function calls capabilities, using a local OpenAI API server on top of Llama_CPP and a model supporting function calls.

Process:
1. Initialize a Llama instance with a specific GPT-based model (e.g., Trelis/Mistral-7B-Instruct-v0.1 with function-calling capabilities), setting the model chat format to 'llama2_functionary' and enabling embeddings.
2. Start the local OpenAI API server on localhost at port 8000 using the Llama instance.
3. Configure the AutoGen setup to use the local API server, including custom function tools for 'weather' and 'traffic' information retrieval.
4. Create a UserProxyAgent representing a human user in the chat, with a system message indicating a human participant.
5. Create an AssistantAgent as a general-purpose chatbot, configured with the local Llama model and the ability to call custom functions.
6. The UserProxyAgent registers custom lambda functions for 'weather' and 'traffic' to simulate responses based on location and/or date.
7. Initiate a chat between the UserProxyAgent and the AssistantAgent with an initial message querying the weather in Tokyo.

This script exemplifies the integration of function-calling capabilities in a chatbot environment, showcasing how custom functionalities can be embedded within an AutoGen agent setup using a local OpenAI API server.
"""

llm = Llama(
    # path to a gguf model supporting function calls (eg. HuggingFace's Trelis/Mistral-7B-Instruct-v0.1-function-calling-v2)
    "/Users/blav/.cache/lm-studio/models/Trelis/Mistral-7B-Instruct-v0.1-function-calling-v2/Mistral-7B-Instruct-v0.1-function-calling-v2.gguf",

    # model chat format (see Llama docs for more info)
    chat_format="llama-2-functionary", 
)

start_openai_api_server(
    llm=llm,
    host="localhost", 
    port=8000,
)

# autogen configuration to use the local API server
llm_config={
    "cache_seed": None,
    "config_list": [{
        "base_url": "http://localhost:8000/v1",
        "model": "dontcare",
        "api_key": "dontcare",
    }],
    "tools": [{
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get weather information for a location.",
            "parameters": {
                "type": "object",
                "title": "weather",
                "properties": {
                    "location": {
                        "title": "location",
                        "type": "string"
                    },
                },
                "required": [ 
                    "location", 
                ]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "traffic",
            "description": "Get traffic information for a location and date.",
            "parameters": {
                "type": "object",
                "title": "traffic",
                "properties": {
                    "location": {
                        "title": "location",
                        "type": "string"
                    },
                    "date": {
                        "title": "date",
                        "type": "string"
                    },
                },
                "required": [ 
                    "location", 
                    "date",
                ]
            }
        }
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

user_proxy.register_function(
    function_map={
        "weather": lambda location: f"weather is nice".format(location=location),
        "traffic": lambda location, date: f"busy".format(location=location, date=date)
    }
)

user_proxy.initiate_chat(
    assistant,
    message="how's the weather in Tokyo today?",
)
from llama_cpp_openai._api_server import start_openai_api_server
from llama_cpp import Llama

"""
This script demonstrates how to set up and start an API server using the Llama library,
which interfaces with an OpenAI-like model for generating text completions and embeddings.
The server mimics the OpenAI API format, allowing for easy integration with systems
already using OpenAI's API endpoints.

First, it initializes a Llama instance with a specified model and configuration.
The `Llama` class requires the path to the GGUF model, the chat format, and a flag 
indicating the use of embeddings.

The `start_openai_api_server` function then launches an API server on the localhost
at the specified port. This server provides endpoints for text completions and embeddings,
similar to the OpenAI API.

The server runs on a separate thread, allowing the main program to continue running
or perform other tasks. The script concludes by joining the thread, which ensures
that the script keeps running as long as the server is active.

Endpoints:
- Text Completions: http://localhost:8000/v1/chat/completions
- Embeddings: http://localhost:8000/v1/embeddings

These endpoints can be used in the same manner as the corresponding OpenAI API endpoints.
"""

llm = Llama(
    # path to gguf model
    "/Users/blav/.cache/lm-studio/models/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q5_K_M.gguf",

    # model chat format (see Llama docs for more info)
    chat_format="mistrallite", 

    # needed by embeddings endpoint
    embedding=True,
)

thread, _ = start_openai_api_server(
    llm=llm,
    host="localhost", 
    port=8000,
)

thread.join()

# Now you can send requests to the API server at 
# http://localhost:8000/v1/chat/completions and http://localhost:8000/v1/embeddings
# using the same format as the OpenAI API.

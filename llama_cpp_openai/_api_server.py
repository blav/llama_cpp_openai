from typing import List, Optional, Literal, Dict, Union
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from threading import Event
from contextlib import asynccontextmanager


import threading
import uvicorn

"""
This module provides functionality for handling chat completions and embeddings requests using FastAPI, Pydantic, and 
the Llama library. It includes classes for defining the structure of requests and responses, and functions for setting 
up and running an OpenAI API-like server.

Classes:
- ChatCompletionsRequestToolFunctionParametersProperty: Represents a property within the parameters of a tool function 
  in a chat completions request.
- ChatCompletionsRequestToolFunctionParameters: Defines the parameters for a tool function used in a chat completions 
  request.
- ChatCompletionsRequestToolFunction: Describes a function tool within a chat completions request.
- ChatCompletionsRequestTool: Represents a tool in a chat completions request.
- ChatCompletionsRequestMessageFunctionCall: Represents a function call within a chat message.
- ChatCompletionsRequestMessage: Represents a single message in a chat completions request.
- ChatCompletionsRequest: Defines the structure of a request for chat completions.
- EmbeddingsRequest: Represents a request for generating embeddings.

Functions:
- completions_endpoint: Asynchronous endpoint for handling chat completions requests.
- embeddings_endpoint: Asynchronous endpoint for handling embeddings requests.
- start_openai_api_server: Starts a thread running an OpenAI API server using FastAPI.

The module integrates various technologies, including FastAPI for web server functionality, Pydantic for data validation 
and serialization, and Llama for machine learning computations. It is designed to create a server capable of processing 
chat completion and embeddings requests similar to the OpenAI API, using structured models for input and output data.
"""

class ChatCompletionsRequestToolFunctionParametersProperty(BaseModel):
    """
    Represents a single property within the parameters of a tool function in a chat completions request.

    Attributes:
    - title (str): The title of the property.
    - type (str): The data type of the property.
    """

    title: str
    type: str

class ChatCompletionsRequestToolFunctionParameters(BaseModel):
    """
    Defines the parameters for a tool function used in a chat completions request.

    Attributes:
    - type (Literal["object"]): Specifies that the parameter type is an object.
    - title (str): The title of the parameters object.
    - required (List[str]): A list of names of required parameters.
    - properties (Dict[str, ChatCompletionsRequestToolFunctionParametersProperty]): A dictionary mapping 
      parameter names to their properties.
    """    
    type: Literal["object"]
    title: str
    required: List[str]
    properties: Dict[str, ChatCompletionsRequestToolFunctionParametersProperty]

class ChatCompletionsRequestToolFunction(BaseModel):
    """
    Describes a function tool within a chat completions request.

    Attributes:
    - name (str): The name of the function tool.
    - description (Optional[str]): An optional description of the function tool.
    - parameters (ChatCompletionsRequestToolFunctionParameters): The parameters for the function tool.
    """
    name: str
    description: Optional[str] = None
    parameters: ChatCompletionsRequestToolFunctionParameters

class ChatCompletionsRequestTool(BaseModel):
    """
    Represents a tool in a chat completions request.

    Attributes:
    - type (Literal["function"]): Indicates that the tool is a function.
    - function (ChatCompletionsRequestToolFunction): The function tool description.
    """
    type: Literal["function"]
    function: ChatCompletionsRequestToolFunction

class ChatCompletionsRequestMessageFunctionCall(BaseModel):
    """
    Represents a function call within a chat message.

    Attributes:
    - name (str): The name of the function being called.
    - arguments (str): The arguments passed to the function call.
    """
    name: str
    arguments: str

class ChatCompletionsRequestMessage(BaseModel):
    """
    Represents a single message in a chat completions request.

    Attributes:
    - content (Optional[str]): The content of the message. Can be None.
    - role (str): The role associated with the message (e.g., 'user', 'system').
    - name (Optional[str]): An optional name associated with the message.
    - function_call (Optional[ChatCompletionsRequestMessageFunctionCall]): An optional function call associated 
      with the message.
    """
    content: Optional[str] = None
    role: str
    name: Optional[str] = None
    function_call: Optional[ChatCompletionsRequestMessageFunctionCall] = None

class ChatCompletionsRequest(BaseModel):
    """
    Defines the structure of a request for chat completions.

    Attributes:
    - messages (List[ChatCompletionsRequestMessage]): A list of messages involved in the chat completion request.
    - model (str): The model to be used for generating chat completions.
    - tools (Optional[List[ChatCompletionsRequestTool]]): An optional list of tools to be used in the chat completion 
      request.
    """
    messages: List[ChatCompletionsRequestMessage]
    model: str
    tools: Optional[List[ChatCompletionsRequestTool]] = None

class EmbeddingsRequest(BaseModel):
    """
    Represents a request for generating embeddings.

    Attributes:
    - model (str): The model to be used for generating embeddings.
    - input (Union[str, List[str]]): The input data for which embeddings are to be generated. Can be a single string 
      or a list of strings.
    - encoding_format (Optional[str]): An optional encoding format for the embeddings.
    """
    model: str
    input: Union[str, List[str]]    
    encoding_format: Optional[str] = None

async def completions_endpoint(llm: Llama, request: ChatCompletionsRequest):
    """
    Asynchronous endpoint for handling chat completions requests.

    This function processes a chat completions request, making adjustments to the request data as necessary, and then 
    calls the appropriate method on the Llama (llm) instance to generate chat completions.

    Args:
    - llm (Llama): An instance of the Llama class, used to generate chat completions.
    - request (ChatCompletionsRequest): The request object containing details for the chat completion. This object 
      should be an instance of a model derived from BaseModel, containing chat messages and optionally tools.

    Returns:
    - The response from the Llama instance's create_chat_completion method, which contains the generated chat completions.

    The function first performs a model dump of the request, excluding any None values. It then restores any 'content' 
    fields in the messages that were suppressed by the model dump. Finally, it calls the Llama instance's 
    create_chat_completion method with the processed messages and tools.
    """

    request = request.model_dump(exclude_none=True)

    # restore None content suppressed by model_dump
    messages = request["messages"]
    for message in messages:
        if not "content" in message:
            message["content"] = None

    return llm.create_chat_completion(
        messages=messages,
        tools=request["tools"] if "tools" in request else None,
    )

async def embeddings_endpoint(llm: Llama, request: EmbeddingsRequest):
    """
    Asynchronous endpoint for handling embeddings requests.

    This function calls the Llama (llm) instance to generate embeddings based on the provided request.

    Args:
    - llm (Llama): An instance of the Llama class, used to generate embeddings.
    - request (EmbeddingsRequest): The request object containing the input data for which embeddings are to be generated.
      The request should be an instance of a model derived from BaseModel, containing the model name and the input data.

    Returns:
    - The response from the Llama instance's create_embedding method, which contains the generated embeddings.

    The function simply calls the Llama instance's create_embedding method with the input data and model specified in the
    request.
    """
    return llm.create_embedding(
        input=request.input,
        model=request.model,
    )



def start_openai_api_server(llm: Llama, host: str = "localhost", port: int = 8000):
    """
    Starts a thread running an OpenAI API server using FastAPI.

    This function creates a FastAPI application with endpoints for handling chat completions and embeddings requests.
    It runs the FastAPI application in a separate thread and waits until the server is ready before returning.

    Args:
    - llm (Llama): An instance of the Llama class, which is used to process the chat completions and embeddings requests.
    - host (str, optional): The hostname on which the FastAPI server will listen. Defaults to "localhost".
    - port (int, optional): The port on which the FastAPI server will listen. Defaults to 8000.

    Returns:
    - Tuple[Thread, FastAPI]: A tuple containing the thread running the FastAPI server and the FastAPI app instance.

    The FastAPI application defines two POST endpoints:
    1. "/v1/chat/completions": Accepts requests in the format of `ChatCompletionsRequest` and uses the Llama instance 
       to create chat completions.
    2. "/v1/embeddings": Accepts requests in the format of `EmbeddingsRequest` and uses the Llama instance to create 
       embeddings.

    The server runs in a daemon thread, ensuring that it does not block the main program from exiting.
    """
    server_ready = Event()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        server_ready.set()
        yield  

    app = FastAPI(lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def completions(request: ChatCompletionsRequest):
        return await completions_endpoint(llm, request)

    @app.post("/v1/embeddings")
    async def embeddings(request: EmbeddingsRequest):
        return await embeddings_endpoint(llm, request)
    
    thread = threading.Thread(
        daemon=True, 
        target=lambda: uvicorn.run(
            app, 
            host=host, 
            port=port, 
        ),
    )
    
    thread.start()
    server_ready.wait()
    return (thread, app)

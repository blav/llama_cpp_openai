# Llama_CPP OpenAI API Server Project Overview

## Introduction
The `llama_cpp_openai` module provides a lightweight implementation of an OpenAI API server on top of
Llama CPP models. This implementation is particularly designed for use with Microsoft AutoGen and includes support for function calls. The project is structured around the `llama_cpp_python` module and is aimed at facilitating the integration of AI models in applications using OpenAI clients or API.

## Project Structure
The project is organized into several key directories and files:

- **llama_cpp_openai**: Contains the core implementation of the API server.
    - `__init__.py`: Initialization file for the module.
    - `_api_server.py`: Defines the OpenAPI server, using FastAPI for handling requests.
    - `_llama_cpp_functions_chat_handler.py`: Implements the `llama-2-functionary` chat handler that supports function calling.

- **examples**: Provides example scripts demonstrating the usage of the API server.
    - `README.md`: Overview and description of example scripts.
    - `autogen_basic.py`: Basic integration of AutoGen with Llama_CPP using the OpenAI API server.
    - `autogen_functions.py`: Sets up an AutoGen chatbot with function calls capabilities.
    - `basic.py`: Demonstrates the setup and start of an API server using the Llama library.

## Key Features
- **FastAPI Integration**: Utilizes FastAPI for efficient and easy-to-use API endpoints.
- **Llama Library Usage**: Leverages the Llama library for handling AI model interactions.
- **Function Call Support**: Includes capabilities for function calls in chatbot environments.
- **Examples for Quick Start**: Provides example scripts for easy understanding and implementation.

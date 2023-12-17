# AutoGen with Llama_CPP and OpenAI API

## Overview
This module demonstrates the integration of Microsoft AutoGen with the Llama_CPP library and OpenAI API server. It includes scripts for setting up an AutoGen chatbot environment and starting a local OpenAI API server using the Llama library, which interfaces with OpenAI-like models.

## Files Description
1. **basic.py**: 
   - A simple script to set up and start an OpenAPI API server on top of Llama_CPP.
   - Mimics the OpenAI API format for compatibility with existing OpenAPI clients.
   - Requires specifying a model and configuration for the `Llama` class.

2. **autogen_basic.py**: 
   - Demonstrates the basic integration of Microsoft AutoGen with Llama_CPP using the OpenAI API server.
   - Initialises a Llama instance with a GGUF model and starts a local OpenAI API server.
   - Runs a basic AutoGen agent

3. **autogen_functions.py**: 
   - Sets up an AutoGen chatbot environment with function call capabilities.
   - Utilizes a local OpenAI API server on top of Llama_CPP.
   - Ideal for models supporting function calls, such as Trelis/Mistral-7B-Instruct-v0.1.
   - Includes initialization of a Llama instance with a specific GPT-based model.

## Setup and Usage
1. In the project folder, run `poetry install`
2. Run the desired script:
   - For setting up a simple API server: `poetry run python basic.py`
   - For basic integration with AutoGen: `poetry run python autogen_basic.py`
   - For a chatbot environment with function calls: `poetry run python autogen_functions.py`


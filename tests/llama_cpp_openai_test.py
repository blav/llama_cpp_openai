import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from llama_cpp_openai import start_openai_api_server  # Assuming your code is in main.py

@pytest.fixture
def mock_llama():
    llama = MagicMock()
    llama.create_chat_completion.return_value = {"result": "chat completion"}
    llama.create_embedding.return_value = {"result": "embedding"}
    return llama

@pytest.fixture
def client(mock_llama):
    with patch("llama_cpp_openai._server.Llama", return_value=mock_llama):
        _, app = start_openai_api_server(mock_llama)
        client = TestClient(app)
        yield client
        # Add any necessary teardown steps here

def test_chat_completions_endpoint(client, mock_llama):
    request_data = {"messages": [{"role": "user", "content": "Hello"}], "model": "test_model"}
    response = client.post("/v1/chat/completions", json=request_data)
    
    assert response.status_code == 200
    assert response.json() == {"result": "chat completion"}
    mock_llama.create_chat_completion.assert_called_once_with(messages=request_data["messages"], tools=None)

def test_embeddings_endpoint(client, mock_llama):
    request_data = {"input": "test input", "model": "test_model"}
    response = client.post("/v1/embeddings", json=request_data)
    
    assert response.status_code == 200
    assert response.json() == {"result": "embedding"}
    mock_llama.create_embedding.assert_called_once_with(input=request_data["input"], model=request_data["model"])

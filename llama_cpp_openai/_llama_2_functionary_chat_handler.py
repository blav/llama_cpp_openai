from llama_cpp import llama, llama_types
from llama_cpp.llama_chat_format import register_chat_completion_handler
from typing import List, Optional, Union, Iterator, Dict
import json

@register_chat_completion_handler("llama-2-functionary")
def llama2_functary_chat_completion_handler(
    llama: llama.Llama,
    messages: List[llama_types.ChatCompletionRequestMessage],
    functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
    function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
    tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    stream: bool = False,
    stop: Optional[Union[str, List[str]]] = [],
    response_format: Optional[llama_types.ChatCompletionRequestResponseFormat] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    tfs_z: float = 1.0,
    mirostat_mode: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    model: Optional[str] = None,
    logits_processor: Optional[llama.LogitsProcessorList] = None,
    grammar: Optional[llama.LlamaGrammar] = None,
    **kwargs,  # type: ignore
) -> Union[llama_types.ChatCompletion, Iterator[llama_types.ChatCompletionChunk]]:
    """
    Handles chat completions for the 'llama2_functionary' plugin in the llama_cpp module.

    This function processes chat completion requests, formats the necessary tools and messages,
    and generates responses based on the given parameters. It's used to enhance chat interactions
    with various functionalities like function calls and tool integrations.

    Parameters:
    - llama (llama.Llama): The Llama instance to interact with.
    - messages (List[llama_types.ChatCompletionRequestMessage]): A list of messages involved in the chat completion.
    - functions (Optional[List[llama_types.ChatCompletionFunction]]): Optional list of functions available for chat completion.
    - function_call (Optional[llama_types.ChatCompletionRequestFunctionCall]): Optional function call details.
    - tools (Optional[List[llama_types.ChatCompletionTool]]): Optional list of tools available for chat completion.
    - tool_choice (Optional[llama_types.ChatCompletionToolChoiceOption]): Optional choice of tool.
    - Various parameters controlling the completion process like temperature, top_p, etc.
    
    Returns:
    Union[llama_types.ChatCompletion, Iterator[llama_types.ChatCompletionChunk]]: The chat completion response.
    """    
    def _format_tools(
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    ) -> str:
        if tools is None or len(tools) == 0:
            return ""
        
        trelis_tools = [{
            "function": tool["function"]["name"],
            "description": tool["function"]["description"],
            "arguments": [{
                "name": name,
                "type": property["type"],
            } for name, property in tool["function"]["parameters"]["properties"].items()],
        } for tool in tools]

        return "<FUNCTIONS>" + json.dumps(trelis_tools, indent=4) + "</FUNCTIONS>\n\n"

    def _format_messages(messages: List[llama_types.ChatCompletionRequestMessage]) -> str:
        lines = []
        state = None
        def _reset_state(): 
            nonlocal state
            state = {
                "user": None,
                "system": None,
                "assistant": None,
            }

        def _push_state(role: str):
            if role is not None and state[role] is None:
                return
            
            if all([v is None for v in state.values()]):
                return
            
            system = ""
            if "system" in state and state["system"] is not None:            
                system = f"<<SYS>>\n{state['system']}\n<</SYS>>\n\n"

            prompt = ""
            if "user" in state and state["user"] is not None:
                prompt = f"[INST] {system}{state['user']} [/INST]"

            if "assistant" in state and state["assistant"] is not None:
                prompt = f"<s>{prompt} {state['assistant']}</s>"

            lines.append(prompt)
            _reset_state()

        def _format_message_content(message) -> str:
            def _format_function_call(function_call):
                return json.dumps({ 
                    "function": function_call["name"], 
                    "arguments": json.loads(function_call["arguments"]) 
                })
            
            if message["role"] == "assistant" and "tool_calls" in message:
                return _format_function_call(message["tool_calls"][0]["function"])
            if message["role"] == "assistant" and "function_call" in message:
                return _format_function_call(message["function_call"])
            elif message["role"] == "function":
                return f"Here is the response to that function call:\n\n\"{message['content']}\""
            return message["content"]
        
        _reset_state()
        i = iter(messages)
        try:
            while True:
                message = next(i)
                if message["role"] == "user" or message["role"] == "function":
                    _push_state("user")
                    state["user"] = _format_message_content(message)
                elif message["role"] == "system":
                    _push_state("system")
                    state["system"] = _format_message_content(message)
                elif message["role"] == "assistant":
                    _push_state("assistant")
                    state["assistant"] = _format_message_content(message)
                else:
                    raise ValueError(f"Unknown role: {message['role']}")
        except StopIteration:
            state = _push_state(role=None)

        return "\n".join(lines)
        
    prompt = _format_tools(messages, functions, tools) + _format_messages(messages)

    assert stream == False, "streaming is not supported yet"
    completion: llama_types.Completion = llama.create_completion(
        prompt=prompt,
        stop=["user:", "</s>"],
        stream=False,
        grammar=grammar,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        typical_p=typical_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repeat_penalty=repeat_penalty,
        tfs_z=tfs_z,
        mirostat_mode=mirostat_mode,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
        model=model,
        logits_processor=logits_processor,
    )

    def _parse_tool_calls(completion: llama_types.Completion):
        content = completion["choices"][0]["text"]
        try:
            function_call = json.loads(content)
            return [{
                "id": function_call["function"],
                "type": "function",
                "function": {
                    "name": function_call["function"],
                    "arguments": json.dumps(function_call["arguments"], indent=2),
                }
            }]
        except json.JSONDecodeError:
            return None
    
    message = {
        "role": "assistant",
    }

    tool_calls = _parse_tool_calls(completion)
    if tool_calls is not None:
        message["content"] = None
        message["tool_calls"] = tool_calls
        message["function_call"] = tool_calls[0]["function"]
    else:
        message["content"] = completion["choices"][0]["text"]

    return llama_types.CreateChatCompletionResponse(
        id="chat" + completion["id"],
        object="chat.completion",
        created=completion["created"],
        model=completion["model"],
        choices=[{
            "index": 0,
            "message": message,
            "finish_reason": "stop",
        }],
        usage=completion["usage"],
    )


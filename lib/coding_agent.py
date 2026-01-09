import json
from typing import Generator, Literal, Optional, Callable, Any, Dict, List, Tuple
import re
from .logger import logger, log_tool_call
from .prompts import *
import base64
from .tools import execute_tool
from .sandbox import BaseSandbox
from .model_config import get_model_config, ModelConfig
from .memory.integration import inject_memories, extract_observation, should_checkpoint
from .hooks import HookManager, HookEvent, HookContext, HookResult

# Global hook manager instance - can be customized per session
_global_hook_manager: Optional[HookManager] = None


def get_hook_manager() -> HookManager:
    """Get or create the global hook manager."""
    global _global_hook_manager
    if _global_hook_manager is None:
        _global_hook_manager = HookManager()
    return _global_hook_manager


def set_hook_manager(manager: HookManager) -> None:
    """Set a custom hook manager."""
    global _global_hook_manager
    _global_hook_manager = manager

# IPython is optional (only needed for notebook environments)
try:
    from IPython.display import Image, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
    Image = None
    display = None


TOKEN_LIMIT = 60_000
COMPRESS_THRESHOLD = 0.7
STATE_SNAPSHOT_PATTERN = re.compile(
    r"<state_snapshot>(.*?)</state_snapshot>", re.DOTALL
)


def clean_messages_for_llm(messages: List[dict]) -> List[dict]:
    return [{k: v for k, v in msg.items() if not k.startswith("_")} for msg in messages]


def compress_messages(client, messages: List[dict], model_config: ModelConfig) -> List[dict]:
    """Compress messages using the compression model from the model config."""
    response = client.responses.create(
        model=model_config.compression_model,
        input=[
            {"role": model_config.system_role, "content": SYSTEM_PROMPT_COMPRESS_MESSAGES},
            *messages,
            {
                "role": "user",
                "content": "First, reason in your scratchpad. Then, generate the <state_snapshot>.",
            },
        ],
    )

    text = response.output_text
    # we extract the <state_snapshot>
    context = "\n".join(STATE_SNAPSHOT_PATTERN.findall(text))
    new_messages = [
        {
            "role": "user",
            "content": f"This is snapshot of the conversation so far:\n{context}",
        },
        {
            "role": "assistant",
            "content": "Got it. Thanks for the additional context!",
        },
    ]

    return new_messages


def format_messages(messages: List[dict]) -> str:
    content = ""
    for message in messages:
        if "role" in message:
            if message["role"] == "user":
                content += f"[user]: {message['content']}\n"
            elif message["role"] == "assistant":
                content += f"[assistant]: {message['content']}\n"
        elif "type" in message:
            if message["type"] == "function_call":
                content += f"[assistant] Calls {message['name']}\n"
            elif message["type"] == "function_call_output":
                content += f"[function_result]: {message['output']}\n"
    return content


def get_compress_message_index(messages: List[dict]) -> int:
    # couting the number of chars
    chars = [len(json.dumps(message)) for message in messages]
    total_chars = sum(chars)
    # we keep a portion of them
    target_chars = total_chars * COMPRESS_THRESHOLD
    curr_chars = 0
    for index, char in enumerate(chars):
        curr_chars += char
        if (curr_chars) >= target_chars:
            return index
    return len(messages)


def get_first_user_message_index(messages: List[dict]) -> int:
    first_user_message_index = 0
    for index, message in enumerate(messages):
        if "role" in message:
            if message["role"] == "user":
                first_user_message_index = index
                break
    return first_user_message_index


def maybe_compress_messages(
    client, messages: List[dict], usage: int, model_config: ModelConfig
) -> List[dict]:
    """Compress messages if token usage exceeds threshold."""
    if usage <= TOKEN_LIMIT * COMPRESS_THRESHOLD:
        return messages
    compress_index = get_compress_message_index(messages)
    if compress_index >= len(messages):
        return messages
    compress_index += get_first_user_message_index(messages[compress_index:])
    if compress_index <= 0:
        return messages
    # edge case, if we cut and the last message is `function_call`
    # we need to add the output as well
    last_message = messages[compress_index - 1]
    if "type" in last_message:
        if last_message["type"] == "function_call":
            # add its output as well
            compress_index += 1

    to_compress_messages = messages[:compress_index]
    to_keep_messages = messages[compress_index:]

    if len(to_compress_messages) > 0:
        logger.info(f"[agent] 📦 compressing messages [0...{compress_index}]...")
        return [*compress_messages(client, to_compress_messages, model_config), *to_keep_messages]

    return messages


def coding_agent(
    client,
    sbx: BaseSandbox,
    query: str,
    tools: Dict[str, Callable],
    tools_schemas: List[dict],
    max_steps: int = 5,
    system: Optional[str] = "You are a senior python programmer",
    messages: Optional[List[dict]] = None,
    usage: Optional[int] = 0,
    model: str = "gpt-4.1-mini",
    memory_manager: Optional[Any] = None,
    checkpoint_interval: int = 100,
    hook_manager: Optional[HookManager] = None,
    session_id: Optional[str] = None,
    **model_kwargs,
) -> Generator[Tuple[dict, dict, int], None, Tuple[List[dict], int]]:
    """
    Core agent loop supporting multiple LLM providers via LiteLLM.

    Args:
        client: LLM client with .responses.create() interface
        sbx: Sandbox for code execution
        query: User query
        tools: Dictionary of tool name -> callable
        tools_schemas: Tool schemas for the LLM
        max_steps: Maximum agent loop iterations
        system: System prompt
        messages: Conversation history
        usage: Token usage counter
        model: Model name (supports OpenAI, Anthropic, Gemini)
        memory_manager: Optional MemoryManager for persistent memory
        checkpoint_interval: Steps between checkpoints (default: 100)
        hook_manager: Optional HookManager for lifecycle hooks
        session_id: Optional session identifier for hooks
        **model_kwargs: Additional model parameters
    """
    # Get model configuration for dynamic system role and compression
    model_config = get_model_config(model)

    # Use provided hook manager or global one
    hooks = hook_manager or get_hook_manager()

    if messages is None:
        messages = []
    # up to here
    start_index = len(messages)

    # Trigger UserPromptSubmit hook
    prompt_hook_result = hooks.trigger(
        HookEvent.UserPromptSubmit,
        query=query,
        messages=messages,
        session_id=session_id,
    )

    # Check if hook wants to block (e.g., content filtering)
    if prompt_hook_result.block:
        logger.warning(f"[hooks] User prompt blocked: {prompt_hook_result.reason}")
        # Return early with a blocked message
        blocked_msg = {
            "role": "assistant",
            "content": f"Request blocked: {prompt_hook_result.reason}"
        }
        messages.append(blocked_msg)
        yield blocked_msg, messages, usage
        return messages, usage

    user_message = {"role": "user", "content": query}
    messages.append(user_message)
    yield user_message, messages, usage

    steps = 0
    # continue till max_steps
    while steps < max_steps:
        # Inject relevant memories from long-term storage
        messages_for_llm = clean_messages_for_llm(messages)
        if memory_manager:
            try:
                relevant_memories = memory_manager.retrieve_context(query, top_k=10)
                if relevant_memories:
                    messages_for_llm = inject_memories(messages_for_llm, relevant_memories)
            except Exception as e:
                logger.warning(f"Failed to retrieve memories: {e}")

        messages_for_llm = maybe_compress_messages(
            client, messages_for_llm, usage, model_config
        )
        response = client.responses.create(
            model=model,
            input=[
                {"role": model_config.system_role, "content": system},
                *messages_for_llm,
            ],
            tools=tools_schemas,
            **model_kwargs,
        )
        usage = response.usage.total_tokens
        has_function_call = False
        for part in response.output:
            messages.append(part.to_dict())
            yield part.to_dict(), messages, usage
            if part.type == "function_call":
                has_function_call = True
                name = part.name
                arguments = part.arguments

                # Parse arguments to dict for hook context (arguments from LLM is usually a JSON string)
                try:
                    args_for_hook = json.loads(arguments) if isinstance(arguments, str) else arguments
                except (json.JSONDecodeError, TypeError):
                    args_for_hook = {}

                # Trigger PreToolUse hook
                pre_tool_result = hooks.trigger(
                    HookEvent.PreToolUse,
                    tool_name=name,
                    arguments=args_for_hook if isinstance(args_for_hook, dict) else {},
                    messages=messages,
                    session_id=session_id,
                )

                # Check if hook wants to block the tool execution
                if pre_tool_result.block:
                    logger.warning(f"[hooks] Tool '{name}' blocked: {pre_tool_result.reason}")
                    result = {"error": f"Tool blocked: {pre_tool_result.reason}"}
                    metadata = {"blocked_by_hook": True}
                else:
                    # Apply any argument modifications from hooks
                    if pre_tool_result.modified_arguments:
                        # Convert modified arguments back to JSON string for execute_tool
                        arguments = json.dumps(pre_tool_result.modified_arguments)
                        logger.debug(f"[hooks] Tool arguments modified by hook")

                    result, metadata = execute_tool(name, arguments, tools, sbx=sbx)

                # Trigger PostToolUse hook
                # Parse arguments to dict if it's a string (for hook context)
                args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments
                post_tool_result = hooks.trigger(
                    HookEvent.PostToolUse,
                    tool_name=name,
                    arguments=args_dict if isinstance(args_dict, dict) else {},
                    result=result,
                    messages=messages,
                    session_id=session_id,
                )

                # Apply any result modifications from hooks
                if post_tool_result.modified_result is not None:
                    result = post_tool_result.modified_result
                    logger.debug(f"[hooks] Tool result modified by hook")

                result_msg = {
                    "type": "function_call_output",
                    "call_id": part.call_id,
                    "output": json.dumps(result),
                    "_metadata": metadata,
                }
                messages.append(result_msg)
                yield result_msg, messages, usage

                # Store observation in memory
                if memory_manager:
                    try:
                        result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                        observation = extract_observation(name, result_str, arguments)
                        memory_manager.add_memory(observation, memory_type="observation", tool_name=name)
                    except Exception as e:
                        logger.warning(f"Failed to store observation: {e}")

        steps += 1

        # Periodic checkpointing
        if memory_manager and should_checkpoint(steps, checkpoint_interval):
            try:
                memory_manager.checkpoint(
                    step=steps,
                    messages=messages,
                    task=query[:200],
                    progress=f"Step {steps}/{max_steps}",
                )
                logger.info(f"[agent] 💾 Checkpoint saved at step {steps}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")

        if not has_function_call:
            break

    # Trigger Stop hook when agent finishes
    hooks.trigger(
        HookEvent.Stop,
        messages=messages,
        session_id=session_id,
        metadata={"total_steps": steps, "usage": usage},
    )

    return messages, usage


def log(generator_func, *args, **kwargs):
    """Wraps the coding_agent and handles logging like the original"""
    gen = generator_func(*args, **kwargs)
    step = 0
    pending_tool_calls = {}  # call_id -> (name, arguments)

    try:
        while True:
            part_dict, messages, usage = next(gen)
            part_type = part_dict.get("type")

            if part_type == "reasoning":
                if step == 0:
                    logger.info(f"✨: [agent-#{step}] Thinking...")
                    step += 1
                logger.info(" ...")
            elif part_type == "message":
                content = part_dict.get("content")
                if content and content[0].get("text"):
                    logger.info(f"✨: {content[0]['text']}")
            elif part_type == "function_call":
                call_id = part_dict.get("call_id")
                name = part_dict.get("name")
                arguments = part_dict.get("arguments")
                pending_tool_calls[call_id] = (name, arguments)
            elif part_type == "function_call_output":
                call_id = part_dict.get("call_id")
                if call_id in pending_tool_calls:
                    name, arguments = pending_tool_calls.pop(call_id)
                    result = json.loads(part_dict.get("output", "{}"))
                    log_tool_call(name, arguments, result)
                metadata = part_dict.get("_metadata")
                if metadata:
                    images = metadata.get("images")
                    if images and HAS_IPYTHON:
                        for image in images:
                            display(Image(data=base64.b64decode(image)))
                    elif images:
                        logger.info(f"[agent] Generated {len(images)} image(s)")

    except StopIteration as e:
        messages, final_usage = e.value
        logger.info(f"[agent] 🔢 tokens: {final_usage} total")
        return messages, final_usage

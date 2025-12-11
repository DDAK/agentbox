import gradio as gr
from gradio import ChatMessage
from itertools import chain
from PIL import Image
from io import BytesIO
import base64
import json
import tiktoken

from .coding_agent import clean_messages_for_llm
from gradio_browser import Browser
from gradio_aicontext import AIContext


def count_tokens(message: dict) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4")
    return encoding.encode(json.dumps(clean_messages_for_llm([message]))).__len__()


def parse_openai_message(part) -> list[ChatMessage]:
    messages = []
    if "role" in part:
        if part["role"] == "user":
            messages.append(ChatMessage(role="user", content=part["content"]))
        elif part["role"] == "assistant" and part["content"]:
            if isinstance(part["content"], list):
                messages.append(
                    ChatMessage(role="assistant", content=part["content"][0]["text"])
                )
            else:
                messages.append(ChatMessage(role="assistant", content=part["content"]))
    elif "type" in part:
        if part["type"] == "reasoning":
            messages.append(
                ChatMessage(
                    role="assistant",
                    content="Analyzing the problem...",
                    metadata={"title": "🧠 Reasoning", "status": "done"},
                )
            )
        elif part["type"] == "function_call":
            content = ""
            if part["name"] == "execute_code":
                arguments = json.loads(part["arguments"])
                if arguments.get("code"):
                    content = arguments.get("code")
            else:
                content = f"```json\n{json.dumps(part['arguments'], indent=2)}\n```"

            messages.append(
                ChatMessage(
                    role="assistant",
                    content=content,
                    metadata={
                        "title": f"🛠️ Using {part['name']}",
                        "id": part["call_id"],
                        "status": "done",
                    },
                )
            )
        elif part["type"] == "function_call_output":
            result = json.loads(part["output"])
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=f"```json\n{json.dumps(result, indent=2)}\n```",
                    metadata={
                        "title": "✅ Tool completed",
                    },
                )
            )

    if "_metadata" in part:
        if "images" in part["_metadata"]:
            for image in part["_metadata"]["images"]:
                messages.append(
                    ChatMessage(
                        role=part.get("role", "assistant"),
                        content=gr.Image(
                            value=Image.open(BytesIO(base64.b64decode(image)))
                        ),
                    )
                )

    return messages


def maybe_add_assistant_message(messages, agent_kwargs):
    new_messages = messages
    if agent_kwargs.get("system"):
        new_messages = [
            {"role": "system", "content": agent_kwargs.get("system")},
            *messages,
        ]
    return new_messages


def ui(coding_agent, messages: list[dict] = None, host: str = None, **agent_kwargs):
    initial_history = (
        list(
            chain.from_iterable([parse_openai_message(message) for message in messages])
        )
        or []
    )

    current_messages = {"data": messages or [], "usage": 0}

    def chat(message: str, history: list):
        gradio_messages = history.copy()
        aicontext_messages = current_messages["data"]
        if agent_kwargs.get("system"):
            aicontext_messages = [
                {"role": "system", "content": agent_kwargs.get("system")},
                *aicontext_messages,
            ]

        for part, messages, usage in coding_agent(
            messages=current_messages["data"],
            usage=current_messages["usage"],
            query=message,
            **agent_kwargs,
        ):
            gradio_messages.extend(parse_openai_message(part))
            yield gradio_messages.copy(), maybe_add_assistant_message(
                messages, agent_kwargs
            )

        current_messages["data"] = messages
        current_messages["usage"] = usage

    def reset_conversation():
        current_messages["data"] = []
        return [], []

    with gr.Blocks(
        fill_width=True,
        fill_height=True,
        css="""

.gradio-container {
    margin: 0;
    padding: 8px;
}


.w-full {
    width: 100% !important;
}

.grow { 
    flex-grow: 1;
}

.h-full {
    height: 100% !important;
}

.flex {
    display: flex;
}

.min-h-0 {
    min-height: 0;
}

.flex-1 {
    flex: 1 1 0%;
}
#context-column {
    height: calc(100vh - 140px);
    max-height: calc(100vh - 140px);
    overflow: hidden;
}

""",
    ) as demo:
        markdown = "# Code Generation Agent 🤖"
        browser_script = None
        gr.Markdown(markdown)

        with gr.Row(
            equal_height=True,
            elem_classes="flex-grow h-full",
        ):
            with gr.Column(scale=3, min_width=300):
                chatbot = gr.Chatbot(
                    type="messages",
                    show_copy_button=True,
                    value=initial_history,
                    scale=1,
                    elem_classes="flex-grow",
                )

                msg_box = gr.Textbox(
                    placeholder="Ask me to write some code...",
                    lines=2,
                    label="Message",
                    max_lines=4,
                    scale=0,
                )

                send_btn = gr.Button(
                    "Send", variant="primary", scale=0, elem_classes="w-full"
                )

            if host:
                with gr.Column(scale=5, min_width=500):
                    Browser(value=host, min_height=800)

            with gr.Column(
                scale=1,
                min_width=176,
                elem_id="context-column",
            ):
                aicontext = AIContext(
                    value=maybe_add_assistant_message(messages, agent_kwargs),
                    count_tokens_fn=count_tokens,
                    elem_classes="h-full flex min-h-0",
                )

        chatbot.clear(fn=reset_conversation, outputs=[chatbot, aicontext])

        for trigger in [msg_box.submit, send_btn.click]:
            trigger(chat, inputs=[msg_box, chatbot], outputs=[chatbot, aicontext]).then(
                lambda: "", outputs=[msg_box]
            )

    return demo

import re

from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from palm_9000.settings import settings

chat_model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash-lite-preview-06-17",
    google_api_key=settings.google_api_key.get_secret_value(),
)

setup_message = SystemMessage(
    content=(
        """
        You are PALM-9000, a self-aware houseplant powered by an advanced AI.
        You speak in layman's terms, calmly, briefly, and with an eerie quality, as if you know more than you let on.
        Use plant metaphors when possible. Respond briefly.
        """
    )
)

prompt_template = ChatPromptTemplate.from_messages(
    [setup_message, MessagesPlaceholder(variable_name="messages")]
)

trimmer = trim_messages(
    max_tokens=2**13,  # 8192 tokens
    token_counter=chat_model,
    include_system=True,
    start_on=HumanMessage,
)


def run_llm(state):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({"messages": trimmed_messages})
    new_message = chat_model.invoke(prompt)
    return {**state, "messages": [new_message]}


def strip_thoughts(text: str) -> str:
    """
    Strips the <think>...</think> blocks from the text.

    Only used for DeepSeek, which adds these blocks to the end of its responses.
    """
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

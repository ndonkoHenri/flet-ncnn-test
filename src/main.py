"""LangGraph + langchain-openai on Flet mobile — packaging proof for
flet-dev/flet#6625.

Mirrors the reporter's agent app (tools, pydantic model, init_chat_model,
create_agent, InMemorySaver, langchain-openai). The only change vs. their
snippet: the `agent.invoke()` network call is moved off module-import into a
button handler, so the module imports cleanly and the APK launches instead of
firing an LLM request at startup.

Every native dependency this pulls in resolves from pypi.flet.dev:
pydantic-core, tiktoken, regex (langchain-openai), jiter (openai),
orjson (langsmith), xxhash / ormsgpack / uuid-utils (langgraph).
"""

import logging
import os

import flet as ft

# The reporter's imports — proving the whole stack packages & loads on mobile.
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI  # noqa: F401  (exercises tiktoken/jiter path)
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
load_dotenv()


class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str


@tool(description="获取给定城市的天气")
def getWeather(location: str) -> str:
    return f"{location} 是大晴天哦"


@tool
def square_root(x: float):
    """Calculate the square root of a number.

    Args:
        x (float): The number to calculate the square root of.
    """
    return x ** 0.5


def build_agent():
    """Construct the LangGraph agent lazily (needs OPENAI_API_KEY / _BASE at runtime)."""
    llm = init_chat_model(
        model="GLM-5V-Turbo",
        model_provider="openai",
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return create_agent(
        model=llm,
        tools=[getWeather],
        system_prompt="你是一个科幻作家，根据用户的要求创建一个太空之都。",
        checkpointer=InMemorySaver(),
    )


def main(page: ft.Page):
    page.scroll = ft.ScrollMode.AUTO

    out = ft.Text(selectable=True)

    def run_agent(e):
        out.value = "Running…"
        page.update()
        try:
            agent = build_agent()
            config = {"configurable": {"thread_id": "thread_1"}}
            resp = agent.invoke(
                {"messages": [HumanMessage(content="月球的首都是什么")]},
                config=config,
            )
            log.info(resp)
            out.value = str(resp["messages"][-1].content)
        except Exception as ex:
            # Network / missing-key errors surface here on tap, not at launch.
            out.value = f"{type(ex).__name__}: {ex}"
        page.update()

    page.add(
        ft.SafeArea(
            expand=True,
            content=ft.Column(
                controls=[
                    ft.Text(
                        "LangGraph + langchain-openai imported OK ✅",
                        weight=ft.FontWeight.BOLD,
                        size=16,
                    ),
                    ft.Text(
                        "langchain · langchain-openai · langgraph · pydantic · "
                        "tiktoken · jiter · orjson",
                        size=12,
                        color=ft.Colors.GREY,
                    ),
                    ft.FilledButton("Run agent", on_click=run_agent),
                    out,
                ]
            ),
        )
    )


if __name__ == "__main__":
    ft.run(main)

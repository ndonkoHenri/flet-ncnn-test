import flet as ft

import langchain
import langchain_core
import langgraph
import pydantic
import pydantic_core
import langgraph.graph


def main(page: ft.Page):
    page.horizontal_alignment = page.vertical_alignment = "center"
    page.add(
        ft.SafeArea(
            content=ft.Text("It works..."),
        )
    )


if __name__ == "__main__":
    ft.run(main)

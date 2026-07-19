import flet as ft


def main(page: ft.Page):
    page.add(
        ft.SafeArea(
            expand=True,
            content=ft.Text(
                "Hello...",
                weight=ft.FontWeight.BOLD,
                size=20,
            ),
        )
    )


if __name__ == "__main__":
    ft.run(main)

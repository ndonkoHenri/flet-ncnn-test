import flet as ft


def main(page: ft.Page):
    page.add(
        ft.SafeArea(
            content=ft.Column(
                spacing=12,
                controls=[],
            ),
        )
    )


if __name__ == "__main__":
    ft.run(main)

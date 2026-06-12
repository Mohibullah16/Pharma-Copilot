from gradio_pharmacopilot_demo import APP_THEME, CSS, demo


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", css=CSS, theme=APP_THEME, allowed_paths=["/data/images"])

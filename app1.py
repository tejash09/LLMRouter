# app.py
import gradio as gr
from router import LLMRouter
import asyncio

# Initialize the LLMRouter with your API key
api_key = "fresed-68HD6scSiwKOCvTXnF94lkEQbl4oDA"
router = LLMRouter(api_key)

# Asynchronous function to generate text using LLMRouter


async def generate_text(prompt: str) -> str:
    result, model = await router.generate(prompt)
    return result, model


def wrap_async_func(async_func):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res = loop.run_until_complete(async_func(*args, **kwargs))
        loop.close()
        return res
    return wrapper


# Create a synchronous wrapper function for the generate_text function
generate_text_sync = wrap_async_func(generate_text)


def app(prompt: str) -> str:
    text, model_used = generate_text_sync(prompt)
    return f"Model used: {model_used}\nGenerated text: {text}"


# Define Gradio interface
iface = gr.Interface(fn=app,
                     inputs="textbox",
                     outputs="textbox",
                     title="Language Model Generator",
                     description="Enter a prompt to generate text using a selected GPT model.")

iface.launch()

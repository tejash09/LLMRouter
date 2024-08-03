import gradio as gr
import asyncio
from router import LLMRouter 
from gtts import gTTS
import tempfile
import os
from openai import OpenAI

api_key = "fresed-68HD6scSiwKOCvTXnF94lkEQbl4oDA"
router = LLMRouter(api_key)

clienta = OpenAI(
    base_url="https://api.naga.ac/v1",
    api_key="ng-adyLEQCrdHyqOSSVjQBoNBqdWHcG2"
)

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        print(f"Error in text-to-speech conversion: {str(e)}")
        return None

def transcribe(audio):
    if audio is None:
        return ""
    try:
        audio_file = open(audio, "rb")
        transcript = clienta.audio.transcriptions.create(model='whisper-large', file=audio_file)
        return transcript.text
    except Exception as e:
        return f"An error occurred during transcription: {str(e)}"

async def chat(message, history):
    try:
        generated_text, model_name = await router.generate(message)
        audio_file = text_to_speech(generated_text)
        history.append((message, generated_text))
        return history, audio_file, model_name
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return history, None, "Error"

iface = gr.Blocks()

with iface:
    gr.Markdown("# LLM Router Chat Interface")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message here")
    audio_input = gr.Audio(type="filepath", label="Or speak your message")
    audio_output = gr.Audio(label="AI Response (Audio)")
    model_used = gr.Textbox(label="Model Used")
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        return asyncio.run(chat(message, chat_history))

    msg.submit(respond, [msg, chatbot], [chatbot, audio_output, model_used])
    clear.click(lambda: None, None, chatbot, queue=False)
    
    audio_input.change(
        fn=transcribe, 
        inputs=[audio_input], 
        outputs=[msg]
    )

iface.launch()
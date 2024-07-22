# app.py
from flask import Flask, render_template, request, jsonify
from router import LLMRouter
import asyncio

app = Flask(__name__)

api_key = "fresed-68HD6scSiwKOCvTXnF94lkEQbl4oDA"
router = LLMRouter(api_key)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json['prompt']

    # Run the async generate function in a synchronous context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result, model = loop.run_until_complete(router.generate(prompt))
    loop.close()

    return jsonify({'result': result, 'model': model})


if __name__ == '__main__':
    app.run(debug=True)

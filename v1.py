import asyncio
from openai import AsyncOpenAI
import time
from typing import List, Dict, Tuple

class LLMRouter:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url='https://fresedgpt.space/v1'
        )
        self.models: List[Dict] = [
            {
                "name": "gpt-3.5-turbo",
                "cost_per_token": 0.0001,
                "avg_response_time": 1.0,
                "complexity_threshold": 0.3
            },
            {
                "name": "gpt-4",
                "cost_per_token": 0.00015,
                "avg_response_time": 0.8,
                "complexity_threshold": 0.6
            },
            {
                "name": "gpt-4-32k",
                "cost_per_token": 0.0002,
                "avg_response_time": 0.6,
                "complexity_threshold": 1.0
            }
        ]
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            print("Spacy model not available. Using fallback complexity calculation.")
            self.nlp = None

    def calculate_complexity(self, prompt: str) -> float:
        if self.nlp:
            return self._calculate_complexity_with_spacy(prompt)
        else:
            return self._calculate_complexity_fallback(prompt)

    def _calculate_complexity_with_spacy(self, prompt: str) -> float:
        doc = self.nlp(prompt)
        
        total_words = len([token for token in doc if not token.is_punct])
        unique_words = len(set([token.text.lower() for token in doc if not token.is_punct]))
        lexical_diversity = unique_words / total_words if total_words > 0 else 0

        depths = [token.dep_.count('>') for token in doc]
        avg_depth = sum(depths) / len(depths) if depths else 0
        max_depth = max(depths) if depths else 0

        named_entities = len(doc.ents)

        sentence_lengths = [len([token for token in sent if not token.is_punct]) for sent in doc.sents]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

        normalized_lexical_diversity = min(lexical_diversity, 1.0)
        normalized_avg_depth = min(avg_depth / 5, 1.0)
        normalized_max_depth = min(max_depth / 10, 1.0)
        normalized_named_entities = min(named_entities / 10, 1.0)
        normalized_avg_sentence_length = min(avg_sentence_length / 30, 1.0)

        complexity = (
            normalized_lexical_diversity * 0.2 +
            normalized_avg_depth * 0.2 +
            normalized_max_depth * 0.2 +
            normalized_named_entities * 0.2 +
            normalized_avg_sentence_length * 0.2
        )

        return complexity

    def _calculate_complexity_fallback(self, prompt: str) -> float:
        words = prompt.split()
        total_words = len(words)
        unique_words = len(set(words))
        avg_word_length = sum(len(word) for word in words) / total_words if total_words > 0 else 0
        
        lexical_diversity = unique_words / total_words if total_words > 0 else 0
        normalized_avg_word_length = min(avg_word_length / 10, 1.0)

        complexity = (lexical_diversity * 0.5 + normalized_avg_word_length * 0.5)
        return complexity

    def select_model(self, prompt: str) -> Dict:
        complexity = self.calculate_complexity(prompt)
        
        for model in self.models:
            if complexity <= model['complexity_threshold']:
                return model

        return self.models[-1]

    async def call_api(self, model: Dict, prompt: str) -> Tuple[str, float]:
        start_time = time.time()
        
        stream = await self.client.chat.completions.create(
            model=model['name'],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        generated_text = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                generated_text += chunk.choices[0].delta.content

        end_time = time.time()
        response_time = end_time - start_time

        return generated_text, response_time

    async def generate(self, prompt: str) -> str:
        selected_model = self.select_model(prompt)
        
        generated_text, response_time = await self.call_api(selected_model, prompt)

        selected_model['avg_response_time'] = (selected_model['avg_response_time'] + response_time) / 2

        return generated_text

async def main():
    api_key = "fresed-68HD6scSiwKOCvTXnF94lkEQbl4oDA"
    router = LLMRouter(api_key)
    prompt = "Translate the following English text to French: 'Hello, how are you?'"

    try:
        result = await router.generate(prompt)
        print(f"Generated text: {result}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    asyncio.run(main())


import json
import os
import time

from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class GroqDeepEvalJudge(DeepEvalBaseLLM):
    def __init__(self, model="openai/gpt-oss-120b", interval=60):
        self.model_name = model
        self.interval = interval
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.last_request = None

    def load_model(self):
        return self.client

    def get_model_name(self):
        return self.model_name

    def generate(self, prompt: str, schema=None):
        if self.last_request is not None:
            elapsed = time.monotonic() - self.last_request
            time.sleep(max(0, self.interval - elapsed))

        self.last_request = time.monotonic()

        messages = [{"role": "user", "content": prompt}]
        if schema is not None:
            messages.insert(0, {
                "role": "system",
                "content": (
                    "Return only a valid JSON object matching this schema:\n"
                    + json.dumps(schema.model_json_schema())
                ),
            })

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
        )

        content = response.choices[0].message.content

        if schema is not None:
            return schema.model_validate_json(content)

        return content

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

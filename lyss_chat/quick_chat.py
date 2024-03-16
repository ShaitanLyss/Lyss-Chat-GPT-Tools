from ast import mod
from datetime import datetime
from openai import OpenAI


import sys
from dotenv import load_dotenv
import os
import argparse
from pydantic import BaseModel
from typing import TYPE_CHECKING, Any, Dict, Literal

from commons import base_dir

if TYPE_CHECKING:
    base_dir: str


class Message(BaseModel):
    content: str
    role: Literal["user"] | Literal["system"] | Literal["assistant"] = "assistant"
    date: datetime = datetime.now()


class Cache(BaseModel):
    messages: Dict[str, Message] = {}

    def has(self, key: str) -> bool:
        return key in self.messages

    def get(self, key: str) -> Message:
        return self.messages[key]

    def set(self, key: str, value: Message) -> None:
        self.messages[key] = value

    def save(self) -> None:
        path = os.path.join(base_dir, "cache.json")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(os.path.join(base_dir, "cache.json"), "w") as f:
            f.write(self.model_dump_json())

    @staticmethod
    def load() -> "Cache":
        try:
            with open(os.path.join(base_dir, "cache.json"), "r") as f:
                data = f.read()
                if not data:
                    return Cache()
                return Cache.model_validate_json(data)
        except FileNotFoundError:
            return Cache()


def chat_with_gpt(message):
    cache = Cache.load()

    if cache.has(message):
        answer = cache.get(message)
        # use cache up until one month
        if (datetime.now() - answer.date).days < 30:
            print(answer.content)
            return

    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Make short straight to the point answers while remaining exhaustive and mentionning important points.",
            },
            {"role": "user", "content": message},
        ],
        stream=True,
    )

    response_contents = []

    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            response_contents.append(content)
            sys.stdout.write(content)
            sys.stdout.flush()
    print()

    cache.set(message, Message(content="".join(response_contents)))
    cache.save()


def main():
    try:
        parser = argparse.ArgumentParser(description="Chat with GPT")
        parser.add_argument(
            "message", type=str, help="The message to chat with GPT", nargs="+"
        )
        args = parser.parse_args()

        chat_with_gpt(" ".join(args.message))
    except KeyboardInterrupt:
        print("\nInterrupted")


if __name__ == "__main__":
    main()

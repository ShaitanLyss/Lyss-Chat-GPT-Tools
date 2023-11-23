import openai
import sys
from dotenv import load_dotenv
import os
import argparse

def chat_with_gpt(message):
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv('OPENAI_API_KEY')

    openai.api_key = api_key

    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=message,
        max_tokens=100,
        temperature=0.7,
        n=1,
        stream=True
    )

    for chunk in response:
        sys.stdout.write(chunk['text'])
        sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description='Chat with GPT')
    parser.add_argument('message', type=str, help='The message to chat with GPT')
    args = parser.parse_args()

    chat_with_gpt(args.message)

if __name__ == '__main__':
    main()

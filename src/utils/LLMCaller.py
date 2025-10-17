import json
from openai import OpenAI
import asyncio
import tiktoken
import time
import os

class LLMCaller:
    def __init__(self):
        try:
            with open("./src/conf/conf.json", "r", encoding="utf-8") as file:
                conf = json.load(file) 
        except FileNotFoundError:
            print("Configuration file './src/conf/conf.json' is not found")
        self.model = conf['model']
        self.temperature = float(conf['temperature'])
        self.stream = conf['stream'].lower() == 'true'
        self.max_tries = int(conf['max_tries'])

        self.total_tokens_used = 0  
        self.input_tokens = 0 
        self.output_tokens = 0
        self.semaphore = asyncio.Semaphore(10)
        api_key = os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('OPENAI_BASE_URL')

        if self.model == 'qwen':
            self.model = "qwen-max-2025-01-25" 
        elif self.model == 'gemini':
            self.model = "gemini-2.5-pro"
        elif self.model == 'gpt':
            self.model = "gpt-4o-2024-08-06"
        elif self.model == 'ds':
            self.mode = "deepseek-r1"
        else:
            raise ValueError(f"Model {self.model} not recognized. Available models: 'qwen', 'gpt', 'ds'")
        
        self.client = OpenAI(
            api_key=api_key, 
            base_url=base_url,
            timeout = 40*60
        )

    def call(self, query):
        tries = 0
        while tries < self.max_tries:
            try:
                if not self.stream:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=query,
                        temperature = self.temperature
                    )
                    if response.usage:
                        self.total_tokens_used += response.usage.total_tokens
                        self.input_tokens += response.usage.prompt_tokens
                        self.output_tokens += response.usage.completion_tokens
                    return response.choices[0].message.content
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=query,
                        temperature=self.temperature,
                        stream=True
                    )

                    full_response = ""
                    for chunk in response:
                        if hasattr(chunk, 'choices') and chunk.choices is not None and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                full_response += delta.content
                                # yield delta.content
                    # print(full_response)

                    encoding = tiktoken.encoding_for_model("gpt-4o")
                    input_token = len(encoding.encode("".join([f"{m['role']}: {m['content']}\n" for m in query])))
                    output_token = len(encoding.encode(full_response))
                    self.total_tokens_used += input_token + output_token
                    self.input_tokens += input_token
                    self.output_tokens += output_token

                    return full_response
            except Exception as e:
                tries += 1
                print(f"Error occurred while processing query: {e}. Retrying ({tries}/{self.max_tries})...")
                time.sleep(1)

        

    async def async_call(self, query):
        async with self.semaphore:
            tries = 0
            while tries < self.max_tries:
                try:
                    if not self.stream:
                        response = await asyncio.to_thread(lambda: self.client.chat.completions.create(model=self.model, messages=query, temperature=self.temperature))
                        
                        if response.usage:
                            self.total_tokens_used += response.usage.total_tokens
                            self.input_tokens += response.usage.prompt_tokens
                            self.output_tokens += response.usage.completion_tokens
                        return response.choices[0].message.content
                    else:
                        response = await asyncio.to_thread(lambda: self.client.chat.completions.create(model=self.model, messages=query, temperature=self.temperature, stream=True))
                        full_response = ""
                        for chunk in response:
                            if hasattr(chunk, 'choices') and chunk.choices is not None and len(chunk.choices) > 0:
                                delta = chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content:
                                    full_response += delta.content

                        encoding = tiktoken.encoding_for_model("gpt-4o")
                        input_token = len(encoding.encode("".join([f"{m['role']}: {m['content']}\n" for m in query])))
                        output_token = len(encoding.encode(full_response))
                        self.total_tokens_used += input_token + output_token
                        self.input_tokens += input_token
                        self.output_tokens += output_token

                        return full_response
                except Exception as e:
                    tries += 1
                    print(f"Error occurred while processing query: {e}. Retrying ({tries}/{self.max_tries})...")
                    await asyncio.sleep(1) 


    async def call_batch_async(self, queries):
        async def delayed_call(query):
            await asyncio.sleep(5)
            return await self.async_call(query)
        tasks = [delayed_call(query) for query in queries]
        results = await asyncio.gather(*tasks)
        return results


    def get_total_tokens_used(self):
        return self.total_tokens_used, self.input_tokens, self.output_tokens


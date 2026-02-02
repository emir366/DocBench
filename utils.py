from openai import OpenAI
import os
from secret_key import OPENAI_API_KEY

def get_gpt_response_openai(text, engine='vllm-nvidia-llama-3-3-70b-instruct-fp8', system_content='Your are a helpful assistant.', json_format=False):
    client = OpenAI(
        base_url=os.environ.get('BASE_URL'), api_key=os.environ.get('CLIENT_API_KEY')
    )
    
    if json_format:
        completion = client.chat.completions.create(
            model=engine,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant designed to output JSON. {system_content}",
                },
                {"role": "user", "content": text},
            ]
        )
    else:
        completion = client.chat.completions.create(
            model=engine,
            messages=[
                {
                    "role": "system",
                    "content": system_content,
                },
                {"role": "user", "content": text},
            ],
        )

    text_response = completion.choices[0].message.content
    # print(text_response)

    response_dict = completion.json()
    # print(response_dict)
    # record(response_dict)
    return text_response

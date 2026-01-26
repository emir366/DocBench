import json
import argparse
import logging
import os
import glob
import requests
from openai import OpenAI
import fitz
import tiktoken
from transformers import AutoTokenizer
from secret_key import OPENAI_API_KEY
from tenacity import retry, stop_after_attempt, wait_random_exponential  # for exponential backoff

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class Runner:
    def __init__(self, system, client, model_name=None):
        self.system = system
        self.client = client
        self.model_name = model_name
    
    @classmethod
    def from_type(cls, system_name, api_base=None, model_name=None):
        client_kwargs = {"api_key": OPENAI_API_KEY}

        # Allow overriding API base (for Ollama or custom endpoints)
        if api_base:
            if not api_base.endswith("/"):
                api_base += "/"
            client_kwargs["base_url"] = api_base
        elif "OPENAI_API_BASE" in os.environ:
            client_kwargs["base_url"] = os.environ["OPENAI_API_BASE"]

        # Create OpenAI client
        client = OpenAI(**client_kwargs)
        return cls(system_name, client, model_name)

    def run(self, folder):
        pdf_path, q_string = self.get_pdfpath_jsonlines_qstr(folder)
        file_content = self.get_document_content(folder, pdf_path)

        if self.system == 'gpt4' or self.system == 'gpt-4o':
            response = self.get_gpt4file_request(file_content, q_string)
        else:
            file_content = self.truncate(file_content)
            response = self.get_gpt_pl_request(file_content, q_string)

        result_dir = f'./data/{folder}/{self.system}_results.txt'
        with open(result_dir, 'w') as f:
            f.write(response)
        return
    
    @staticmethod
    def get_pdfpath_jsonlines_qstr(folder):
        jsonlines = open(f'./data/{folder}/{folder}_qa.jsonl', 'r').readlines()
        pdf_path = glob.glob(f'./data/{folder}/*.pdf')[0]
        q_string = "Based on the uploaded information, answer the following questions. You should answer all above questions line by line with numerical numbers.\n"
        for i, line in enumerate(jsonlines):
            question = json.loads(line)['question']
            q_string += f'{i+1}. {question}\n'
        qstr_dir = f'./data/{folder}/{folder}_qstring.txt'
        with open(qstr_dir, 'w') as f:
            f.write(q_string)
        return pdf_path, q_string
    
    def get_document_content(self, folder, pdf_path):
        if self.system == 'gpt4' or self.system == 'gpt-4o':
            file_content = self.client.files.create(file=open(pdf_path, "rb"), purpose="assistants").id
        else:
            content_dir = f'./data/{folder}/{folder}_content.txt'
            if os.path.exists(content_dir):
                file_content = open(content_dir,'r').read()
            else:
                doc = fitz.open(pdf_path)
                file_content = ""
                image_index = 1
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_blocks = page.get_text("dict")["blocks"]
                    for block in text_blocks:
                        if block["type"] == 0:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    file_content += span["text"]
                        elif block["type"] == 1:
                            file_content += f"[image{image_index}]"
                            image_index += 1
                        file_content += "\n"
                with open(content_dir, 'w') as f:
                    f.write(file_content)
        return file_content
    
    def truncate(self, file_content):
        if self.system == 'gpt-4o_pl':
            encoding = tiktoken.encoding_for_model('gpt-4-turbo')
            file_content = encoding.encode(file_content)
            if len(file_content) > 120000:
                file_content = encoding.decode(file_content[:120000])
        
        elif self.system == 'gpt4_pl':
            # gpt-4 tokenizer is used as a proxy, 8000 is a safe limit for local models
            encoding = tiktoken.encoding_for_model('gpt-4') 
            encoded = encoding.encode(file_content)
            if len(encoded) > 8000:
                file_content = encoding.decode(encoded[:8000])

        elif self.system == 'gpt3.5':
            encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
            encoded = encoding.encode(file_content)
            if len(encoded) > 15000:
                file_content = encoding.decode(encoded[:15000])
        return file_content

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_gpt4file_request(self, file_content, q_string):
        if self.model_name:
            engine = self.model_name
        else:
            if self.system == "gpt4": engine = "gpt-4-turbo"
            else: engine = self.system

        assistant = self.client.beta.assistants.create(
            name="Document Assistant",
            instructions="You are a helpful assistant that helps users answer questions based on the given document.",
            model=engine,
            tools=[{"type": "file_search"}],
        )
        thread = self.client.beta.threads.create(
            messages=[{
                "role": "user", "content": q_string,
                # Attach the new file to the message.
                "attachments": [{ "file_id": file_content, "tools": [{"type": "file_search"}] }],
            }]
        )
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=assistant.id
        )
        messages = list(self.client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
        message_content = messages[0].content[0].text
        annotations = message_content.annotations
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(annotation.text, "")
        return message_content.value

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_gpt_pl_request(self, file_content, q_string):
        model_to_use = self.model_name
        if not model_to_use:
            system2model = {'gpt4_pl': 'gpt-4-turbo', 'gpt-4o_pl': 'gpt-4o', 'gpt3.5_pl': 'gpt-3.5-turbo-0125'}
            model_to_use = system2model.get(self.system, self.system)

        response = self.client.chat.completions.create(
            model=model_to_use,
            messages=[{
                    "role": "system", "content": "You are a helpful assistant that helps users answer questions based on the given document.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Here is the document:\n\n"
                        f"{file_content}\n\n"
                        "Questions:\n"
                        f"{q_string}"
                    )
                }   
            ]
        ).choices[0].message.content
        return response


class Runner_OSS:
    def __init__(self, system, tokenizer):
        self.system = system
        self.tokenizer = tokenizer
    
    @classmethod
    def from_type(cls, system_name, model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return cls(system_name, tokenizer)

    def truncate(self, file_content, max_token_len=127000):
        file_ids = self.tokenizer.encode(file_content)
        if len(file_ids) >= max_token_len:
            file_ids = file_ids[:max_token_len]
        return self.tokenizer.decode(file_ids)

    def get_document_content(self, folder):
        pdf_path = glob.glob(f'./data/{folder}/*.pdf')[0]
        content_dir = f'./data/{folder}/{folder}_content.txt'
        if os.path.exists(content_dir):
            file_content = open(content_dir,'r').read()
        else:
            doc = fitz.open(pdf_path)
            file_content = ""
            image_index = 1
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text_blocks = page.get_text("dict")["blocks"]
                for block in text_blocks:
                    if block["type"] == 0:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                file_content += span["text"]
                    elif block["type"] == 1:
                        file_content += f"[image{image_index}]"
                        image_index += 1
                    file_content += "\n"
            with open(content_dir, 'w') as f:
                f.write(file_content)
        return file_content
    
    def run(self, folder, max_new_tokens=100):
        file_content = self.get_document_content(folder)
        file_content = self.truncate(file_content)
        ori_dict = [json.loads(line) for line in open(f'./data/{folder}/{folder}_qa.jsonl').readlines()]
        q_list = [json.loads(line)['question'] for line in open(f'./data/{folder}/{folder}_qa.jsonl').readlines()]

        if self.system == 'commandr-35b':
            input_prompts = [f'<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>This is the document content.\n{file_content}\nAnswer the question based on the given text: {q} <|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' for q in q_list]
        elif self.system == 'llama2-13b':
            input_prompts = [f"<s>[INST] <<SYS>> \n You are a helpful assistant that helps users answer questions based on the given document.<<</SYS>> {q}<[/INST]" for q in q_list]
        elif self.system == 'llama3-8b' or self.system == 'llama3-70b':
            input_prompts = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant that helps users answer questions based on the given document.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{file_content}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" for q in q_list]
        else:
            input_prompts = [f'This is the document content.\n{file_content}\nAnswer the question based on the given text: {q}' for q in q_list]

        with open(f'./data/{folder}/{self.system}_results.jsonl', 'a') as f:
            for i, q in enumerate(q_list):
                data = {"prompt": input_prompts[i], "model": "my_model", "max_tokens": max_new_tokens, "temperature": 0}
                response = requests.post("http://0.0.0.0:8081" + "/v1/completions", json=data)
                answer = response.json()['choices'][0]['text']
                res_dict = ori_dict[i]
                res_dict['sys_ans'] = answer
                f.write(json.dumps(res_dict) + '\n')
        return input_prompts, q_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, default="gpt-4o", choices=['gpt-4o','gpt4', 'gpt4_pl', 'gpt-4o_pl', 'gpt3.5', 'phi3-medium','commandr-35b','internlm2-20b', 'internlm2-7b', 'chatglm3-6b','gpt3.5','llama3-8b','llama3-70b','yi1.5-9b', 'yi1.5-34b','mixtral-8x7b','mistral-7b','gemma-7b', 'llama2-13b', 'kimi', 'claude3','glm4', 'qwen2.5', 'ernie4'], help="The name of running system.")
    parser.add_argument("--model_dir", type=str, default="YOUR OWN MODEL DIR", help="Downloaded model paths.")
    parser.add_argument("--initial_folder", type=int, default=0, help="From which folder to begin evaluation.")
    parser.add_argument("--total_folder_number", type=int, default=228, help="Total pdf folders.")
    parser.add_argument("--api_base", type=str, default=None, help="Custom API base URL (e.g., http://localhost:11434/v1/).")
    parser.add_argument("--model_name", type=str, default=None, help="Custom model name (e.g., gemma3n:e2b).")

    args = parser.parse_args()
    system = args.system

    if 'gpt' in system:
        runner = Runner.from_type(system, api_base=args.api_base, model_name=args.model_name)
    else:
        runner = Runner_OSS.from_type(system, args.model_dir)

    cur_folder_num = args.initial_folder
    logger.info(f"***** Running evaluation for system = {system} *****")
    
    while cur_folder_num <= args.total_folder_number:
        logger.info(f"Folder = {cur_folder_num}")
        runner.run(cur_folder_num)
        cur_folder_num += 1


if __name__ == "__main__":
    main()

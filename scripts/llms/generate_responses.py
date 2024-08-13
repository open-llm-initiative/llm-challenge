import os
import openai
import anthropic
import torch
import pandas as pd
import platform
import accelerate
import boto3
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from openai import OpenAI
from data_frames import GenerateResponsesDataFrameHandler
from enum import Enum
from botocore.exceptions import ClientError


class HuggingFaceModels(Enum):
    Qwen2_0_5B_Instruct =  "Qwen/Qwen2-0.5B-Instruct" 
    Qwen2_1_5B_Instruct =  "Qwen/Qwen2-1.5B-Instruct"
    Gemma_2_2B =  "google/gemma-2-2b" 
    Qwen2_7B_Instruct = "Qwen/Qwen2-7B-Instruct" 
    Phi_3_small_128k_instruct = "microsoft/Phi-3-small-128k-instruct"

#helper class to simplify pipeline creation and usage.
class CustomChatPipelineHuggingFace:
    def __init__(self, model_name, device="cuda"):
        # Initialize the model and tokenizer
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            token=hf_token,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
        self.device = device

        # Define and set the chat template
        chat_template = """
        <|system|> {system} <|endoftext|>
        <|user|> {user} <|endoftext|>
        <|assistant|>
        """
        self.tokenizer.chat_template = chat_template

    def release(self):
        del self.model
        del self.tokenizer
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
   
    def get_model_name(self):
        return self.model.config._name_or_path

    def __call__(self, prompt, max_new_tokens=340):
        # Create the chat messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # Apply the chat template to the messages
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        #gemma_2_2B doesn't like the chat template.
        if self.model.config._name_or_path == HuggingFaceModels.Gemma_2_2B.value:
            text = prompt

        # Tokenize the input and create an attention mask
        model_inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)
        
        # Generate the response using attention mask
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens
        )
        
        # Strip the input tokens from the generated response
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated response
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] 
        return response

class APIModelsHelper():
    model_call_dictionary = {}

    def get_models_dict(self):
        return self.model_call_dictionary

    def call_model(self, model_name, prompt):
        callable_response_func = self.get_models_dict[model_name][0]
        formatted_prompt = self.get_models_dict[model_name][1](prompt)

        return callable_response_func(model_name, formatted_prompt)

    @staticmethod
    def no_formatter(prompt):
        return f"{prompt}"

    @staticmethod
    def llama_prompt_formatter(prompt):
        formatted_prompt = f"""
            <|begin_of_text|>
            <|start_header_id|>user<|end_header_id|>
            {prompt}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """
        return formatted_prompt 

    @staticmethod
    def generate_response_from_openai(model_name, prompt):
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        response = client.chat.completions.create(
                    model=model_name,
                    temperature = 1,
                    messages=[
                         {"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": prompt}
                ]
        )
        return response.choices[0].message.content

    @staticmethod
    def generate_response_from_anthropic(model_name, prompt):
        client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        response = client.messages.create(
            model=model_name,
            system = "You are a helpful assistant.",
            messages=[
                         {"role": "user", "content": prompt}
                ],
            max_tokens=512,
            stop_sequences=["\n\nHuman:", "\n\nAI:", "END_OF_TEXT"]
        )
        return ''.join([block.text for block in response.content if block.type == 'text'])

    @staticmethod
    def generate_response_from_aws(model_name, prompt):
        client = boto3.client("bedrock-runtime", region_name="us-west-2")
        # Format the request payload using the model's native structure.
        native_request = {
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature": 1,
        }

        if model_name == 'mistral.mistral-large-2407-v1:0':
            del native_request['max_gen_len'] #nuances between models
            native_request['max_tokens'] = 512 

        # Convert the native request to JSON.
        request = json.dumps(native_request)
        
        try:
            # Invoke the model with the request.
            response = client.invoke_model(modelId=model_name, body=request)
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_name}'. Reason: {e}")

        # Decode the response body.
        model_response = json.loads(response["body"].read())

        # Extract and print the response text.
        if model_name == 'mistral.mistral-large-2407-v1:0':
            response_text = model_response["outputs"][0]["text"] #nuances between models
        else:
            response_text = model_response["generation"]
        
        return response_text

    model_call_dictionary['meta.llama3-1-70b-instruct-v1:0'] = [generate_response_from_aws, llama_prompt_formatter]
    model_call_dictionary['mistral.mistral-large-2407-v1:0'] = [generate_response_from_aws, no_formatter]
    model_call_dictionary['claude-3-5-sonnet-20240620'] = [generate_response_from_anthropic, no_formatter]
    model_call_dictionary['claude-3-haiku-20240307'] = [generate_response_from_anthropic, no_formatter]
    model_call_dictionary['gpt-3.5-turbo'] = [generate_response_from_openai, no_formatter]
    model_call_dictionary['gpt-4o'] = [generate_response_from_openai, no_formatter]

def generate_response_from_hugging_face_models():
    return None

def generate_responses_from_api_models(challenge_df, response_df):
    return None

def sanatize_response_to_html():
    return None

def get_df_for_generation():
    return None

if __name__ == '__main__':
    print("Starting repsponse generation. First step: load LLM responses from OSS LLMs on HuggingFace \n")
    challenge_prompt_df = pd.read_csv('../../data/challenge_setup.csv')
    resposne_df = GenerateResponsesDataFrameHandler(challenge_prompt_df)
    gemma_pipeline = CustomChatPipelineHuggingFace(model_name=HuggingFaceModels.Gemma_2_2B.value)

    for index, row in challenge_prompt_df.iterrows():
        prompt_to_llm = row.prompt
        if row.context != "None":
            prompt_to_llm+= " " + row.context
        
        prompt_response = gemma_pipeline(prompt_to_llm)
        prompt_id = gemma_pipeline.get_model_name() + "_" + str(index)
        resposne_df.add(row, prompt_id, prompt_response)
        print("Response:", prompt_id, prompt_response, "\n")

    resposne_df.to_csv()


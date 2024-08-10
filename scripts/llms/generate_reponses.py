import os
import openai
import anthropic
import torch
import pandas as pd
import platform
import accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from openai import OpenAI
from hf_models_enum import HuggingFaceModels

anthropic_models = {
    "Claude 3 Haiku": {"api_key": os.environ['ANTHROPIC_API_KEY'], "model": "claude-3-haiku"},
    "Claude 3.5 Sonnet": {"api_key": os.environ['ANTHROPIC_API_KEY'], "model": "claude-3.5-sonnet"}
}

openai_models = {
    "OpenAI GPT 3.5-Turbo": {"api_key": os.environ['OPENAI_API_KEY'], "model": "gpt-3.5-turbo"},
    "OpenAI GPT4-o": {"api_key": os.environ['OPENAI_API_KEY'], "model": "gpt-4o"}
}


def call_openai_model(model_name, prompt, context):
    client = OpenAI(
            api_key=openai_models[model_name]['api_key'],
    )
    response = client.chat.completions.create(
                model=openai_models[model_name]['model'],
                temperature = 1,
                messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{context}\n\Prompt: {prompt}"}
            ]
    )
    return response.choices[0].message.content

def call_anthropic_model(model_name, prompt, context):
    client = anthropic.Anthropic(api_key = anthropic_models[model_name]['api_key'])
    prompt = f"{context}\n\nQuestion: {prompt}"
    response = client.completions.create(
        model=anthropic_models[model_name]['model'],
        prompt=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{context}\n\Prompt: {prompt}"}
            ],
        max_tokens_to_sample=512,
        stop_sequences=[anthropic.e]
    )
    return response.completion.strip()

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

    def __call__(self, prompt, max_new_tokens=512):
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

if __name__ == '__main__':
    print("Starting repsponse generation. First step: load LLM responses from OSS LLMs on HuggingFace \n")
    challenge_prompt_df = pd.read_csv('../../data/challenge_setup.csv')
    gemma_pipeline = CustomChatPipelineHuggingFace(model_name=HuggingFaceModels.Gemma_2_2B.value)

    for row in challenge_prompt_df.itertuples():
        if row.Index == 2:
            print("Response:", gemma_pipeline(row.prompt + "   " + row.context))


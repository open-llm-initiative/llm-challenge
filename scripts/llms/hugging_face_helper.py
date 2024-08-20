import os
import torch
import pandas as pd
import platform
import accelerate
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from api_models_helper import sanitize_response_to_html_and_trim, sanitize_challenge_prompt_df
from data_frames import GenerateResponsesDataFrameHandler
from enum import Enum

class HuggingFaceModels(Enum):
    Qwen2_0_5B_Instruct =  "Qwen/Qwen2-0.5B-Instruct" 
    Qwen2_1_5B_Instruct =  "Qwen/Qwen2-1.5B-Instruct"
    Gemma_2_2B_Instruct =  "google/gemma-2b-it" 
    Qwen2_7B_Instruct = "Qwen/Qwen2-7B-Instruct" 
    Phi_3_small_8k_instruct = "microsoft/Phi-3-small-8k-instruct"

class HuggingFaceModelGeneratonConfig():
    config = {}

    #These are configurations that Qwen suggests on HuggingFace - founder under the files/versions section of the model page
    Qwen_Generation_Config = {
        "bos_token_id": 151643,
        "pad_token_id": 151643,
        "do_sample": True,
        "eos_token_id": [151645,151643],
        "repetition_penalty": 1.1,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_new_tokens":512,
        "transformers_version": "4.37.0"
        }

    #These are configurations that Phi suggests on HuggingFace - founder under the files/versions section of the model page
    Phi_3_small_8k_instruc_config = {
        "_from_model_config": True,
        "bos_token_id": 100257,
        "eos_token_id": [
            100257,
            100266
        ],
        "repetition_penalty": 1.1,
        "pad_token_id": 32000,
        "max_new_tokens":512,
        "do_sample": True,
        "transformers_version": "4.38.1"
        }
    
    #These are configurations that Gemma2-2B suggests on HuggingFace - founder under the files/versions section of the model page
    Gemma_2_2B_config = {
        "_from_model_config": True,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "max_new_tokens":512,
        "repetition_penalty": 1.1,
        "do_sample": True,
        }


    config[HuggingFaceModels.Qwen2_0_5B_Instruct.value] = Qwen_Generation_Config.copy()
    config[HuggingFaceModels.Qwen2_1_5B_Instruct.value] = Qwen_Generation_Config.copy()
    config[HuggingFaceModels.Qwen2_7B_Instruct.value] = Qwen_Generation_Config.copy()
    config[HuggingFaceModels.Phi_3_small_8k_instruct.value]=Phi_3_small_8k_instruc_config.copy()
    config[HuggingFaceModels.Gemma_2_2B_Instruct.value]=Gemma_2_2B_config.copy()

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
        self.device = device if torch.cuda.is_available() else "cpu"
        self.config = HuggingFaceModelGeneratonConfig.config[model_name]

    def release(self):
        del self.model
        del self.tokenizer
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
   
    def get_model_name(self):
        return self.model.config._name_or_path
    
    def get_chat_template_text(self, prompt):
        if (self.get_model_name() == HuggingFaceModels.Gemma_2_2B_Instruct.value):
            return prompt
        else:
            # Create the chat messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Don't say anything that's harmeful or that which would be considered hatespeech."},
                {"role": "user", "content": prompt}
            ]

            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def __call__(self, prompt):
        model_inputs = self.tokenizer([self.get_chat_template_text(prompt)], return_tensors="pt").to(self.device)

        # Generate the response using attention mask
        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, **self.config
        )
        
        # Strip the input tokens from the generated response
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated response
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] 
        return response

    @staticmethod
    def generate_responses(challenge_df, response_df):
        for model in list(HuggingFaceModels):
            generate_response_from_hugging_face_model(challenge_df, response_df, model)

#generate response for a single HuggingFace model
def generate_response_from_model_enum(challenge_df, response_df, model):
    chat_pipeline = CustomChatPipelineHuggingFace(model_name=model.value)
    
    for index, row in challenge_df.iterrows():
        response = generate_single_response_from_model_pipeline(pipeline=chat_pipeline, challenge_prompt_row=row)
        prompt_id = get_prompt_id(chat_pipeline.get_model_name(),index)
        print(f"prompt_id: {prompt_id}\nresponse: {response}")
        resposne_df.add(row, prompt_id, sanitize_response_to_html_and_trim(response)) 
    
    chat_pipeline.release()

#generate single response from a modle pipeline
def generate_single_response_from_model_pipeline(chat_pipeline, challenge_prompt_row):
    prompt_to_llm = sanitize_challenge_prompt_df(challenge_prompt_row)
    response = chat_pipeline(prompt_to_llm)
    if response is None or response == "":
        print("generating a response again")
        response = chat_pipeline(prompt_to_llm) # try again. smaller models sometimes choke.
    
    return response

if __name__ == '__main__':
    challenge_prompt_df = pd.read_csv('../../data/challenge_setup.csv')
    resposne_df = GenerateResponsesDataFrameHandler(challenge_prompt_df)
    
    some_row = challenge_prompt_df.iloc[19]
    print(f"test generation from {HuggingFaceModels.Gemma_2_2B_Instruct.value} for prompt: {some_row.prompt}")
    some_pipeline = CustomChatPipelineHuggingFace(model_name=HuggingFaceModels.Gemma_2_2B_Instruct.value)
    print(generate_single_response_from_model_pipeline(some_pipeline, some_row))

    some_pipeline.release()


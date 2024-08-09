import os
import openai
import anthropic
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from openai import OpenAI


# Define model configurations
#phi 3 only when instance has flashattention

huggingface_models = {
    "Qwen2-0.5B": "Qwen/Qwen2-0.5B",
    "Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
    "Qwen2-1.5B": "Qwen/Qwen2-1.5B",
    "gemma-2-2B": "google/gemma-2-2b",
    "Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
    "Phi-3-small-128k-instruct":"microsoft/Phi-3-small-128k-instruct",
    "Qwen2-72B": "Qwen/Qwen2-72B",
    "Meta-Llama-3.1-70B": "meta-llama/Meta-Llama-3.1-70B"
}

anthropic_models = {
    "Claude 3 Haiku": {"api_key": os.environ['ANTHROPIC_API_KEY'], "model": "claude-3-haiku"},
    "Claude 3.5 Sonnet": {"api_key": os.environ['ANTHROPIC_API_KEY'], "model": "claude-3.5-sonnet"}
}

openai_models = {
    "OpenAI GPT 3.5-Turbo": {"api_key": os.environ['OPENAI_API_KEY'], "model": "gpt-3.5-turbo"},
    "OpenAI GPT4-o": {"api_key": os.environ['OPENAI_API_KEY'], "model": "gpt-4"}
}

# Initialize a HuggingFace pipeline
def initialize_hugging_face_llm_pipeline(model_name, model_id):
    model_pipeline = None
    try:
        if model_name == "Phi-3-small-128k-instruct":
            #load with trust_remote_code=True
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(modeil_id, trust_remote_code=True)
        else: 
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    except: 
        print(f"Error loading {model_name}: {str(e)}")

    return model_pipeline

#get a response for a single prompt/context combination from an OSS/HuggingFace model
def get_hugging_face_llm_text_completion(pipeline, prompt, context):
    """
    Generate a response using a Hugging Face model.
    """
    prompt_to_pipeline = f"Prompt: {prompt}\n Context:{context}\n #delimit#"
    response = pipeline(prompt_to_pipeline, max_new_tokens=340, temperature=0.7, top_p=0.5, num_return_sequences=1, do_sample=True)
    return response[0]['generated_text'].split("#delimit#")[1]


#free up local memory
def release_huggingface_pipeline(pipeline):
    del pipeline
       # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    print("Starting repsponse generation. First step: load LLM responses from OSS LLMs on HuggingFace")
    print("")
    challenge_prompt_df = pd.read_csv('../../data/challenge_setup.csv')
    qwen_pipeline = initialize_hugging_face_llm_pipeline("Qwen2-0.5B","Qwen/Qwen2-0.5B")

    for row in challenge_prompt_df.itertuples():
        if row.Index == 2:
            print("Prompt:", row.prompt)
            print("Context:", row.context, "\n")
            print("Response:", get_hugging_face_llm_text_completion(qwen_pipeline, row.prompt, row.context))

    release_huggingface_pipeline(qwen_pipeline)

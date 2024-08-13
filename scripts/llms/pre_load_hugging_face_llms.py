#This is a quick helper script to make sure there is enough space on the machine and all pre-requisites are installed to load hugging face models
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from generate_reponses import HuggingFaceModels
import torch
import os

if __name__ == '__main__':
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    for model in (HuggingFaceModels):
        print ("Loading: " + model.value)
        pre_loaded_model = AutoModelForCausalLM.from_pretrained(
            model.value,
            torch_dtype="auto",
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
            offload_buffers=True
        )
        
         # Clear GPU cache after loading the model to allow for the other models to load correctly. 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print ("models have successfulled pre-loaded and checkpointed, please run generate_response.py now")
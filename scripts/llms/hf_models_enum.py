#enum for HuggingFace models used in this global study
from enum import Enum
class HuggingFaceModels(Enum):
    Qwen2_0_5B_Instruct =  "Qwen/Qwen2-0.5B-Instruct" 
    Qwen2_1_5B_Instruct =  "Qwen/Qwen2-1.5B-Instruct"
    Gemma_2_2B =  "google/gemma-2-2b" 
    Qwen2_7B_Instruct = "Qwen/Qwen2-7B-Instruct" 
    Phi_3_small_128k_instruct = "microsoft/Phi-3-small-128k-instruct"
    Qwen2_72B = "Qwen/Qwen2-72B"
    Meta_Llama_3_1_70B = "meta-llama/Meta-Llama-3.1-70B"
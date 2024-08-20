import os
import openai
import pandas as pd
import json

from data_frames import GenerateResponsesDataFrameHandler
from api_models_helper import APIModelsHelper
from hugging_face_helper import CustomChatPipelineHuggingFace

if __name__ == '__main__':
    challenge_prompt_df = pd.read_csv('../../data/challenge_setup.csv')
    resposne_df = GenerateResponsesDataFrameHandler(challenge_prompt_df)

    print("Starting repsponse generation. First step: load LLM responses from OSS LLMs on HuggingFace \n")
    CustomChatPipelineHuggingFace.generate_responses(challenge_df=challenge_prompt_df, response_df=resposne_df)

    print("Starting repsponse generation. Second step: load LLM responses API-based Models \n")
    model_helper = APIModelsHelper()
    model_helper.generate_responses(challenge_df=challenge_prompt_df, response_df=resposne_df)

    resposne_df.to_csv()


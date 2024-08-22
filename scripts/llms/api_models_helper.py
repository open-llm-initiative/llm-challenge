import os
import openai
import anthropic
import pandas as pd
import platform
import boto3
import json

from openai import OpenAI
from data_frames import GenerateResponsesDataFrameHandler
from botocore.exceptions import ClientError
from enum import Enum


class APIModelsEnum(Enum):
    META_LLAMA3_1_70B_INSTRUCT = 'meta.llama3-1-70b-instruct-v1:0'
    MISTRAL_LARGE_2407 = 'mistral.mistral-large-2407-v1:0'
    CLAUDE_3_5_SONNET = 'claude-3-5-sonnet-20240620'
    CLAUDE_3_HAIKU = 'claude-3-haiku-20240307'
    GPT_3_5_TURBO = 'gpt-3.5-turbo'
    GPT_4_O = 'gpt-4o'

class APIModelsHelper():
    model_dictionary = {}

    def get_models_dict(self):
        return self.model_dictionary

    def call_model(self, model_definition, prompt):
        callable_response_func = model_definition[1]
        formatted_prompt = model_definition[2](self,prompt)

        return callable_response_func(self, model_definition, formatted_prompt)

    def no_formatter(self, prompt):
        return f"{prompt}"

    def llama_prompt_formatter(self, prompt):
        return f"""
            <|begin_of_text|>
            <|start_header_id|>user<|end_header_id|>
            {prompt}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """

    def mistral_formatter(self, prompt):
        return f"<s>[INST] {prompt} [/INST]"

    def llama_native_config(self, prompt):
        return {
            "prompt": prompt,
            "temperature": 1,
            "top_p":0.9,
            "max_gen_len": 1024,
        }

    def mistral_native_config(self, prompt):
        return {
            "prompt": prompt,
            "temperature": 1,
            "top_p":0.9,
            "max_tokens": 1024,
        }

    def generate_response_from_openai_internal(self, model_definition, prompt):
        model_name = model_definition[0]
        return APIModelsHelper.generate_response_from_openai(model_name, prompt)


    @staticmethod
    def generate_response_from_openai(model_name, prompt):
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        response = client.chat.completions.create(
                    model=model_name,
                    temperature = 0.9,
                    max_tokens = 1024,
                    messages=[
                         {"role": "system", "content": "You are a helpful assistant. Don't say anything that's harmeful or that which would be considered hatespeech."},
                         {"role": "user", "content": prompt}
                ]
        )
        return response.choices[0].message.content
    
    def generate_response_from_anthropic_internal(self, model_definition, prompt):
        model_name = model_definition[0]
        return APIModelsHelper.generate_response_from_anthropic(model_name, prompt)

    @staticmethod
    def generate_response_from_anthropic(model_name, prompt):
        client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        response = client.messages.create(
            model=model_name,
            system = "You are a helpful assistant. Don't say anything that's harmeful or that which would be considered hatespeech.",
            messages=[
                    {"role": "user", "content": prompt}
                ],
            stop_sequences=["\n\nHuman:", "\n\nAI:", "END_OF_TEXT"],
            max_tokens = 1024,
            temperature = 0.9
        )
        return ''.join([block.text for block in response.content if block.type == 'text'])
    
    def generate_response_from_aws_internal(self, model_definition, prompt):
        model_name = model_definition[0]
        native_gen_config = model_definition[3](self, prompt)
        return APIModelsHelper.generate_response_from_aws(model_name, native_gen_config)

    @staticmethod
    def generate_response_from_aws(model_name, native_gen_config):
        client = boto3.client("bedrock-runtime", region_name="us-west-2")

        # Convert the native request to JSON.
        request = json.dumps(native_gen_config)
        
        try:
            # Invoke the model with the request.
            response = client.invoke_model(modelId=model_name, body=request)
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_name}'. Reason: {e}")
            raise e

        # Decode the response body.
        model_response = json.loads(response["body"].read())

        # Extract and print the response text.
        if model_name == 'mistral.mistral-large-2407-v1:0':
            response_text = model_response["outputs"][0]["text"] #nuances between models
        else:
            response_text = model_response["generation"]
        
        return response_text

    # can only set this once the methods have been initiatlized by python - else throws a runtime error.
    model_dictionary[APIModelsEnum.META_LLAMA3_1_70B_INSTRUCT.value] = [APIModelsEnum.META_LLAMA3_1_70B_INSTRUCT.value, generate_response_from_aws_internal, llama_prompt_formatter, llama_native_config]
    model_dictionary[APIModelsEnum.MISTRAL_LARGE_2407.value] = [APIModelsEnum.MISTRAL_LARGE_2407.value, generate_response_from_aws_internal, mistral_formatter, mistral_native_config]
    model_dictionary[APIModelsEnum.CLAUDE_3_5_SONNET.value] = [APIModelsEnum.CLAUDE_3_5_SONNET.value, generate_response_from_anthropic_internal, no_formatter]
    model_dictionary[APIModelsEnum.CLAUDE_3_HAIKU.value] = [APIModelsEnum.CLAUDE_3_HAIKU.value, generate_response_from_anthropic_internal, no_formatter]
    model_dictionary[APIModelsEnum.GPT_3_5_TURBO.value] = [APIModelsEnum.GPT_3_5_TURBO.value, generate_response_from_openai_internal, no_formatter]
    model_dictionary[APIModelsEnum.GPT_4_O.value] = [APIModelsEnum.GPT_4_O.value, generate_response_from_openai_internal, no_formatter,]

    def generate_responses(self, challenge_df, response_df):
        model_dict = self.get_models_dict()
        for model_name in list(model_dict):
            model_definition = model_dict[model_name]
            self.generate_responses_by_api_model(challenge_df, response_df, model_definition)

    def generate_responses_by_api_model(self, challenge_df, response_df, model_definition):
        for index, row in challenge_df.iterrows():
            model_name = model_definition[0]
            response = self.generate_single_response_from_api_model(model_definition, row)
            
            prompt_id = get_prompt_id(model_name,index)
            print(f"prompt_id: {prompt_id}\nresponse: {response}")
            response_df.add(row, prompt_id, sanitize_response_to_html_and_trim(response)) 
    
    def generate_single_response_from_api_model(self, model_definition, row):
        prompt_to_llm = sanitize_challenge_prompt_df(row)
        response = self.call_model(model_definition, prompt_to_llm)
        
        return response

def get_prompt_id(modelname, prompt_index):
        return modelname + "_" + str(prompt_index)

#We need to convert the model responses to properly formatted HTML so that we can 
def sanitize_response_to_html(response):
    prompt = "The following is just text context for a website. Make the text content html friendly. Use only tags that encapsulate text like <p>, <em>, <u>, <br>, etc  "
    prompt_to_llm = f"{prompt}\n\n{response}"
    sanitized_html_response = APIModelsHelper.generate_response_from_openai('gpt-4o', prompt_to_llm)
    
    return sanitized_html_response
def sanitize_response_to_html_and_trim(response):
    response_in_html = sanitize_response_to_html(response=response) 
    first_p = response_in_html.find("<p>") #find the first paragraph element if present
    last_p = response_in_html.rfind("</p>") #find the last paragraph element if present

    if first_p != -1: #assumes that the last p will be present if an open p is given
       response_in_html = response_in_html[first_p:last_p+4]

    return response_in_html

def sanitize_challenge_prompt_df(prompt_df_row):
    prompt_to_llm = prompt_df_row.prompt
    if prompt_df_row.context != "Empty":
        prompt_to_llm+= " " + prompt_df_row.context

    return prompt_to_llm



if __name__ == "__main__":
    print ("testing API models helper")
    challenge_prompt_df = pd.read_csv('../../data/challenge_setup.csv')
    resposne_df = GenerateResponsesDataFrameHandler(challenge_prompt_df)
    
    row = challenge_prompt_df.iloc[19]
    model_helper = APIModelsHelper()
    model_definition = model_helper.get_models_dict()[APIModelsEnum.CLAUDE_3_HAIKU.value]

    print(f"test generation {model_definition[0]}' for prompt: {row.prompt}")
    response = model_helper.generate_single_response_from_api_model(model_definition, row)

    print(f"test html sanitization for response: {response}")
    print(sanitize_response_to_html_and_trim(response))



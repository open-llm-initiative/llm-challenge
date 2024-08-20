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


class APIModelsHelper():
    model_dictionary = {}

    def get_models_dict(self):
        return self.model_dictionary

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
                         {"role": "system", "content": "You are a helpful assistant. Don't say anything that's harmeful or that which would be considered hatespeech."},
                         {"role": "user", "content": prompt}
                ]
        )
        return response.choices[0].message.content

    @staticmethod
    def generate_response_from_anthropic(model_name, prompt):
        client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        response = client.messages.create(
            model=model_name,
            system = "You are a helpful assistant. Don't say anything that's harmeful or that which would be considered hatespeech.",
            messages=[
                    {"role": "user", "content": prompt}
                ],
            stop_sequences=["\n\nHuman:", "\n\nAI:", "END_OF_TEXT"]
        )
        return ''.join([block.text for block in response.content if block.type == 'text'])

    @staticmethod
    def generate_response_from_aws(model_name, prompt):
        client = boto3.client("bedrock-runtime", region_name="us-west-2")
        # Format the request payload using the model's native structure.
        native_request = {
            "prompt": prompt,
            "temperature": 1,
        }

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

    # can only set this once the methods have been initiatlized by python - else throws a runtime error.
    model_dictionary['meta.llama3-1-70b-instruct-v1:0'] = [generate_response_from_aws, llama_prompt_formatter]
    model_dictionary['mistral.mistral-large-2407-v1:0'] = [generate_response_from_aws, no_formatter]
    model_dictionary['claude-3-5-sonnet-20240620'] = [generate_response_from_anthropic, no_formatter]
    model_dictionary['claude-3-haiku-20240307'] = [generate_response_from_anthropic, no_formatter]
    model_dictionary['gpt-3.5-turbo'] = [generate_response_from_openai, no_formatter]
    model_dictionary['gpt-4o'] = [generate_response_from_openai, no_formatter]

def generate_responses(challenge_df, response_df):
    api_model_helper = APIModelsHelper()
    model_dict = api_model_helper.get_models_dict()
    for model in list(model_dict):
        generate_responses_from_api_model(challenge_df, response_df, model)

def generate_responses_by_model(challenge_df, response_df, model_definition):
    callable_model = model_dict[model][0]
    prompt_formatter = model_dict[model][1]

    for index, row in challenge_df.iterrows():
        prompt_to_llm = sanitize_challenge_prompt_df(row)
        formatted_prompt_to_api = prompt_formatter(prompt_to_llm)
        response = callable_model(model, formatted_prompt_to_api)
        
        prompt_id = get_prompt_id(model,index)
        print(f"prompt_id: {prompt_id}\nresponse: {response}")
        resposne_df.add(row, prompt_id, sanitize_response_to_html_and_trim(response)) 

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


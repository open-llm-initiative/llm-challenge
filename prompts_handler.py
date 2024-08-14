import boto3
import pandas as pd
import math
import random
from datetime import datetime
from botocore.exceptions import ClientError

prompts_table_name = "challenge_prompts"
prompts_table_name_test = "challenge_prompts_test"

challenge_responses_table_name = "challenge_responses"
challenge_respones_table_name_test = "challenge_responses_test"

class PromptDBHandler():
    def __init__(self, prod=False):
        #load all prompts and use them to serve
        self.dynamodb = boto3.resource('dynamodb')
        self.prod = prod
    
    def load_all_prompts_and_return_set(self):
        try:
            table_name = prompts_table_name if self.prod else prompts_table_name_test
            table = self.dynamodb.Table(table_name)
            response_ddb = table.scan()

            loaded_prompts = response_ddb.get('Items', [])
            while 'LastEvaluatedKey' in response_ddb:
                response_ddb = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
                loaded_prompts.extend(response_ddb.get('Items', []))

            prompt_id_set = set()
            for prompt in loaded_prompts:
                prompt_id_set.add(prompt['prompt_id'])

            return prompt_id_set
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't Load All Prompts . Reason: {e}")

    def get_prompt(self, prompt_id):
        try:
            table_name = prompts_table_name if self.prod else prompts_table_name_test
            table = self.dynamodb.Table(table_name)

            response = table.get_item(
                Key={
                    "prompt_id": prompt_id
                }
            )
            item = response['Item']
            return item  
        except Exception as e:
            print(f"Error getting item: {e}")
            return e
    
    def get_random_prompt(self, prompt_set):
        index = random.randint(0, len(prompt_set)-1)
        list_of_prompts = list(prompt_set)
        return self.get_prompt(list_of_prompts[index])

    def load_challenge_responses_in_ddb_rom_csv(self, prod=False):
        prompt_challenge_df = pd.read_csv('data/prompts_with_responses_llms.csv')
        try:
            table_name = prompts_table_name if prod else prompts_table_name_test
            table = self.dynamodb.Table(table_name)
            
            for index, row in prompt_challenge_df.iterrows():
                table.put_item(Item=row.to_dict())

        except Exception as e:
            print(f"Error putting items: {e}")
            return None

    # Function to store an item in the DynamoDB table
    @staticmethod
    def store_challenge_response(self, session_id, prompt_id, rating, timestamp=None, prod=False):
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()  # Generate a timestamp if not provided

        # Create the item to store
        item = {
            'session_id': session_id,
            'prompt_id': prompt_id,
            'rating': rating,
            'timestamp': timestamp
        }

        try:
            table_name = challenge_responses_table_name if prod else challenge_respones_table_name_test
            table = self.dynamodb.Table(table_name)

            # Store the item in DynamoDB
            table.put_item(Item=item)
        except Exception as e:
            print(f"Error saving the response: {e}")
            return e

if __name__ == '__main__':
    db_handler = PromptDBHandler()
    db_handler.load_challenge_responses_in_ddb_rom_csv()
    prompt_id_set = db_handler.load_all_prompts_and_return_set()
    random_prompt = db_handler.get_random_prompt(prompt_id_set)
    print(random_prompt)
    

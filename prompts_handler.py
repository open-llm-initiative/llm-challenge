import boto3
from datetime import datetime
import pandas as pd

prompts_table_name = "challenge_prompts"
prompts_table_name_test = "challenge_prompts_test"

challenge_responses_table_name = "challenge_responses"
challenge_respones_table_name_test = "challenge_responses_test"

class PromptDBHandler():
    def __init__(self, prod=False):
        #load all prompts and use them to serve
        self.dynamodb = boto3.resource('dynamodb')
        self.prod = prod
 
    def get_prompt(self, prompt_id):
        try:
            table_name = prompts_table_name if prod else prompts_table_name_test
            table = self.dynamodb.Table(table_name)

            response = table.get_item(
                TableName="challenge_prompts",
                Key={
                    "prompt_id": {
                        'S': prompt_id
                    }
                }
            )
            item = response['Item']
            return item  
        except Exception as e:
            print(f"Error getting item: {e}")
            return e
    
    @staticmethod
    def load_challenge_responses_from_csv(prod=False):
        prompt_challenge_df = pd.read_csv('data/prompts_with_responses.csv')
        try:
            table_name = prompts_table_name if self.prod else prompts_table_name_test
            table = self.dynamodb.Table(table_name)
            
            for index, row in prompt_challenge_df.iterrows():
                print(row.to_dict())
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
    db_handler.load_challenge_responses_from_csv()

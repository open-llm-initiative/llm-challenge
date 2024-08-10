import boto3

class PromptHandler():
    def __init__(self):
        #load all prompts and use them to serve
        return None

    def __get_sample_prompt(prompt_id):
        dynamodb = boto3.client('dynamodb')
        try:
            response = dynamodb.get_item(
                TableName="challenge_prompts",
                Key={
                    "prompt_id": {
                        'N': prompt_id
                    }
                }
            )
            item = response['Item']
            return item  
        except Exception as e:
            print(f"Error getting item: {e}")
            return None
    
    def get_prompt_to_display(self, uuid):
        return None

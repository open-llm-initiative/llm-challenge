from flask import Flask, render_template
import boto3

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('challenge.html')  

def get_sample(sample_id):
  dynamodb = boto3.client('dynamodb')
  try:
      response = dynamodb.get_item(
          TableName="global_llm_challenge_prompts",
          Key={
              "prompt_id": {
                  'S': partition_key_value
              }
          }
      )
      item = response['Item']
      return item
  except Exception as e:
      print(f"Error getting item: {e}")
      return None

if __name__ == '__main__':
    app.run(debug=True)   
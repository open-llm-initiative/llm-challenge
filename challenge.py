from flask import Flask, render_template, request
import boto3

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('challenge.html', prompt_id="default_1234")  

@app.route('/submit', methods=['POST'])
def submission():
    prompt_id = request.form['prompt_id']
    rating = request.form['rating']

    # Process the data (e.g., save to database)
    # ...

    return render_template('submission.html')

#get sample prompt for the user
def get_sample_prompt(prompt_id):
  dynamodb = boto3.client('dynamodb')
  try:
      response = dynamodb.get_item(
          TableName="challenge_prompts",
          Key={
              "prompt_id": {
                  'N': sample_id
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
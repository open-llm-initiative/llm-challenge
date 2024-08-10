from flask import Flask, render_template, session, request, make_response
import boto3
import string
import random
import os

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv("APP_SECRET_KEY", "A123B456C789D10")

def generate_random_string(length):
  """Generates a random string of specified length"""
  letters_and_digits = string.ascii_letters + string.digits
  result_str = ''.join(random.choice(letters_and_digits) for i in range(length))  

  return result_str

def setSessionCookieForUser():
    if "challenge_cookie" not in session:
        session["challenge_cookie"] = generate_random_string(12)

def getSessionCookieForUser():
    return session["challenge_cookie"]

@app.route('/')
def index():

    setSessionCookieForUser()
    response = make_response(render_template('challenge.html', prompt_id="default_1234"))

    return response

@app.route('/submit', methods=['POST'])
def submission():
    prompt_id = request.form['prompt_id']
    rating = request.form['rating_for_prompt']

    # Process the data (e.g., save to database)
    # ...

    return render_template('submission.html')

if __name__ == '__main__':
    app.run(debug=True)   
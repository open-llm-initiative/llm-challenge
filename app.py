import boto3
import string
import random
import os
import datetime
import base64
import watchtower
import common
from common.prompts_handler import PromptDBHandler
from flask import Flask, render_template, session, request, make_response

app = Flask(__name__, static_folder='static', template_folder='templates')
prompt_db_hanlder = PromptDBHandler()

@app.before_request
def initialize():
    prompt_db_hanlder.load_all_prompts()   

#hsoted under the open-llm-initiative.com - therefore we need a home page for the intiative
@app.route('/')
def index():
    start_time = datetime.datetime.now() #record the time we first showed the page. will use for analysis later.
    response = make_response(get_response_template_with_random_prompt('index.html',start_time))
    set_user_cookie(response)
    
    return response

#route to the callenge
@app.route('/challenge')
def challenge():
    start_time = datetime.datetime.now() #record the time we first showed the page. will use for analysis later.
    response = make_response(get_response_template_with_random_prompt('challenge_index.html',start_time))
    set_user_cookie(response)
    
    return response

#route to a rating submission
@app.route('/challenge/submit', methods=['POST'])
def submission():
    prompt_id = request.form['prompt_id']
    if prompt_id == None:
        start_time = datetime.datetime.now() #record the time we first showed the page. will use for analysis later.
        response = make_response(get_response_template_with_random_prompt('challenge_index.html',start_time))
        return response
    
    #check for valid prompt_id - user might be overiding form, in that case show error message
    prompt_id_set = prompt_db_hanlder.get_prompt_set()
    if prompt_id not in prompt_id_set:
        start_time = datetime.datetime.now()
        response = make_response(get_response_template_with_random_prompt('challenge_error.html', start_time))
        return response
        
    #check if we got a valid rating- user might be overiding form, in that case show error message
    rating = request.form['rating_for_prompt']
    if rating not in {"1","2","3","4","5"}:
        start_time = datetime.datetime.now()
        response = make_response(get_response_template_with_random_prompt('challenge_error.html', start_time))
        return response


    start_time = request.form['time_start']
    # Process the data (e.g., save to database)
    try:
        PromptDBHandler.store_challenge_response(prompt_db_hanlder, session_id=get_user_cookie(), prompt_id=prompt_id, rating=rating, start_time_iso=start_time)
        response = make_response(get_response_template_with_random_prompt('challenge_submission.html'))
    except Exception as e:
        print(f"Error saving the response: {e}")
        return e
        
    return response

#Helper method to render a response template with a new random prompt from DDB
def get_response_template_with_random_prompt(template_name, time_start=None):
    if time_start is None:
        time_start = datetime.datetime.now()  # Generate a timestamp if not provided
    
    prompt = prompt_db_hanlder.get_random_prompt()
    prompt_id = prompt['prompt_id']
    prompt_content = prompt['prompt']
    context_content = prompt['context']
    prompt_response = prompt['generated_response']

    #TODO: fix bug in response generator
    if prompt_response.startswith('```html\n'):
        end_index = prompt_response.rfind('```')-len(prompt_response)
        prompt_response = prompt_response[7:-4]

    return render_template(template_name, prompt_id=prompt_id, prompt=prompt_content, context=context_content, response=prompt_response, time_start=time_start.isoformat())

def generate_random_string(length):
  """Generates a random string of specified length"""
  letters_and_digits = string.ascii_letters + string.digits
  result_str = ''.join(random.choice(letters_and_digits) for i in range(length))  

  return result_str

#Store sesssion cookier for user
def set_user_cookie(response):
    if get_user_cookie() is None:
        response.set_cookie("challenge_cookie", generate_random_string(12), max_age=60*60*24) 

#Get session cookier for user
def get_user_cookie():
    return request.cookies.get('challenge_cookie') 

if __name__ == '__main__':
    prompt_db_hanlder.load_all_prompts()
    app.run(debug=True) 
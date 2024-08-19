import boto3
import string
import random
import os
import datetime
import base64
import watchtower
import common
from common.prompts_handler import PromptDBHandler
from common.logger import CloudWatchLogger
from flask import Flask, render_template, session, request, make_response

logger = None
app_loaded = False
prompt_db_hanlder = None
app = Flask(__name__, static_folder='static', template_folder='templates')

@app.before_request
def initialize():
    global prompt_db_hanlder, logger, app_loaded 
    if not app_loaded:
        is_production_str = os.getenv("PRODUCTION_CHALLENGE", "false")
        is_production = is_production_str.lower() == "true"

        prompt_db_hanlder = PromptDBHandler(is_production)
        logger = CloudWatchLogger(is_production)
        
        prompt_db_hanlder.load_all_prompts() 
        logger.info("Application Successfully Initiatilized")
        app_loaded = True
  
#hosted under the open-llm-initiative.com - therefore we need a home page for the intiative
@app.route('/')
def index():
    start_time = datetime.datetime.now() #record the time we first showed the page. will use for analysis later.
    response = make_response(get_response_template_with_random_prompt('index.html',start_time))
    set_user_cookie(response)

    logger.info("Loaded the main Open LLM Initiative Page")
    return response

#route to the callenge
@app.route('/challenge')
def challenge():
    start_time = datetime.datetime.now() #record the time we first showed the page. will use for analysis later.
    response = make_response(get_response_template_with_random_prompt('challenge_index.html',start_time))
    set_user_cookie(response)

    logger.info(f"Loaded the main challenge page for {get_user_cookie()}")
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
        logger.error(f"No prompt-id set for {get_user_cookie()}")
        return response
        
    #check if we got a valid rating- user might be overiding form, in that case show error message
    rating = request.form['rating_for_prompt']
    if rating not in {"1","2","3","4","5"}:
        start_time = datetime.datetime.now()
        response = make_response(get_response_template_with_random_prompt('challenge_error.html', start_time))
        logger.error(f"No rating-id set for {prompt_id} for user {get_user_cookie()}")
        return response


    start_time = request.form['time_start']
    # Process the data (e.g., save to database)
    try:
        PromptDBHandler.store_challenge_response(prompt_db_hanlder, session_id=get_user_cookie(), prompt_id=prompt_id, rating=rating, start_time_iso=start_time)
        response = make_response(get_response_template_with_random_prompt('challenge_submission.html'))
    except Exception as e:
        logger.error(f"Error saving rating: {e}")
        return e
    
    logger.info(f"Saved rating {rating} for {prompt_id} for user {get_user_cookie()}")
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
    app.run(debug=True) 
import boto3
import string
import random
import os
import datetime
import base64
import watchtower
from flask import Flask, render_template, session, request, make_response
from prompts_handler import PromptDBHandler

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv("APP_SECRET_KEY", "A123B456C789D10")
prompt_db_hanlder = PromptDBHandler()

def generate_random_string(length):
  """Generates a random string of specified length"""
  letters_and_digits = string.ascii_letters + string.digits
  result_str = ''.join(random.choice(letters_and_digits) for i in range(length))  

  return result_str

#Store sesssion cookier for user
def setUserCookie(response):
    if getUserCookie() is None:
        response.set_cookie("challenge_cookie", generate_random_string(12), max_age=60*60*24) 

#Get session cookier for user
def getUserCookie():
    return request.cookies.get('challenge_cookie')

#Get cookie that helps us understand which prompts the user has already seen
def getSeenPromptsCookie():
    return request.cookies.get("seen_prompts")

#Set cookie to the list of prompts seen by the user
def setSeenPromptsSeenCookie(response, current_prompt_id):
    base64_encoded_cookie_str = ""
    if getSeenPromptsCookie():
       base64_encoded_cookie_str = getSeenPromptsCookie()
    
    response.set_cookie("seen_prompts", decode_encode_base64_new_item_in_list_as_string(base64_encoded_cookie_str,current_prompt_id), max_age=60*60*24*7) 

def decode_encode_base64_new_item_in_list_as_string(list_as_encoded_base64_string, new_item):
    base64_decoded_list = base64.b64decode(list_as_encoded_base64_string)
    decoded_list = base64_decoded_list.decode('utf-8').split(',')
    decoded_list.append(str(new_item))
    encoded_new_list = base64.b64encode(', '.join(decoded_list).encode('utf-8'))
    return encoded_new_list.decode('utf-8')

def __test__decode_encode_base64_new_item_in_list_as_string():
    base64_encoded_list = "NCw1LDYsNw=="
    base64_encoded_list_with_new_item = "NCwgNSwgNiwgNywgOA=="
    new_list = __decode_encode_base64_new_item_in_list_as_string(base64_encoded_list, 8)
    if base64_encoded_list_with_new_item == new_list:
        return True
    else: 
        raise Exception 

@app.route('/')
def index():
    start_time = datetime.datetime.now() #record the time we first showed the page. will use for analysis later.
    response = make_response(get_response_template_with_random_prompt('challenge.html',start_time))
    setUserCookie(response)
    
    return response


@app.route('/submit', methods=['POST'])
def submission():
    prompt_id = request.form['prompt_id']
    if prompt_id == None:
        start_time = datetime.datetime.now() #record the time we first showed the page. will use for analysis later.
        response = make_response(get_response_template_with_random_prompt('challenge.html',start_time))
        return response
    
    #TODO: handle bad prompt_ids, or bad ratings, don't let the user mess that up
    rating = request.form['rating_for_prompt']
    start_time = request.form['time_start']

    # Process the data (e.g., save to database)
    try:
        PromptDBHandler.store_challenge_response(prompt_db_hanlder, session_id=getUserCookie(), prompt_id=prompt_id, rating=rating, start_time_iso=start_time)
        response = make_response(get_response_template_with_random_prompt('submission.html'))
        setSeenPromptsSeenCookie(response, prompt_id)
    except Exception as e:
        print(f"Error saving the response: {e}")
        
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

@app.before_request
def initialize():
    prompt_db_hanlder.load_all_prompts()   

if __name__ == '__main__':
    prompt_db_hanlder.load_all_prompts()
    app.run(debug=True) 
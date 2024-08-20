
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from data_frames import GenerateResponsesDataFrameHandler

file_path_test = '../../../data/prompts_with_responses_test.csv'

def test_generate_response_data_frame_handler_to_csv():
    original_df = pd.DataFrame({'challenge': 'yes', 'prompt': 'write me a poem'}, index=[0])
    final_df = GenerateResponsesDataFrameHandler(original_df)
    print("Successfully initiatlized class")

    final_df.add(original_df, "1234", "The sky is blue")
    final_df.add(original_df, "5678", "The sky is green")
    print("Successfully added rows to the class")

    final_df.to_csv(file_path=file_path_test)
    print("Successfully written to file. Now cleaning up")
    
    # Delete the file
    if os.path.exists(file_path_test):
        os.remove(file_path_test)
        print(f"File {file_path_test} has been deleted.")
    else:
        print(f"File {file_path_test} does not exist.")

    print("##End Of Test Case. \n")

def test_generate_response_data_frame_handler():
    original_df = pd.DataFrame({'challenge': 'yes', 'prompt': 'write me a poem'}, index=[0])
    final_df = GenerateResponsesDataFrameHandler(original_df)
    print("Successfully initiatlized class")

    final_df.add(original_df, "1234", "The sky is blue")
    final_df.add(original_df, "5678", "The sky is green")
    print("Successfully added rows to the class")
    print("##End Of Test Case. \n")


if __name__ == '__main__':
    test_generate_response_data_frame_handler()
    test_generate_response_data_frame_handler_to_csv()
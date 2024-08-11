import pandas as pd
import os

#helper class that will maintain the data as we iterate over the responses from LLMs

file_path = '../../data/prompts_with_responses.csv'
file_path_test = '../../data/prompts_with_responses_test.csv'
class GenerateResponsesDataFrameHandler():

    def __init__(self, original_df):
        self.df = pd.DataFrame(columns=original_df.columns)
        self.df['prompt_response_id'] = []
        self.df['generated_response'] = []

    def add(self, df_row, prompt_response_id, prompt_response):
        new_dict = df_row.to_dict()
        new_dict['prompt_response_id'] = prompt_response_id
        new_dict['generated_response'] = prompt_response
        new_df = pd.DataFrame(new_dict, index=[0])
        self.df = pd.concat([self.df, new_df], ignore_index=True)

    def __repr__(self):
        # This method is useful for debugging and provides a detailed string representation
        return f"MyDataFrameClass(df={self.df})"

    def to_csv(self, file_path=file_path):
        self.df.to_csv(file_path, index=False)


def test_generate_response_data_frame_handler():
    original_df = pd.DataFrame({'challenge': 'yes', 'prompt': 'write me a poem'}, index=[0])
    final_df = GenerateResponsesDataFrameHandler(original_df)
    print("Successfully initiatlized class")

    final_df.add(original_df, "1234", "The sky is blue")
    final_df.add(original_df, "5678", "The sky is green")
    print("Successfully added rows to the class")
    print("##End Of Test Case. \n")

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

if __name__ == '__main__':
    test_generate_response_data_frame_handler()
    test_generate_response_data_frame_handler_to_csv()
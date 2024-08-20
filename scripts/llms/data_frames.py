import pandas as pd
import os

#helper class that will maintain the data as we iterate over the responses from LLMs

file_path = '../../data/prompts_with_responses.csv'
class GenerateResponsesDataFrameHandler():

    def __init__(self, original_df):
        self.df = pd.DataFrame(columns=original_df.columns)
        self.df['prompt_id'] = []
        self.df['generated_response'] = []

    def add(self, df_row, prompt_response_id, prompt_response):
        new_dict = df_row.to_dict()
        new_dict['prompt_id'] = prompt_response_id
        new_dict['generated_response'] = prompt_response
        new_df = pd.DataFrame(new_dict, index=[0])
        self.df = pd.concat([self.df, new_df], ignore_index=True)

    def __repr__(self):
        # This method is useful for debugging and provides a detailed string representation
        return f"MyDataFrameClass(df={self.df})"

    def to_csv(self, file_path=file_path):
        self.df.to_csv(file_path, index=False)
import logging
import watchtower

log_group_name = "challenge_prompts_log_group"
log_stream_name = "challenge_log_stream"

#Helper class to add cloud watch logs
class CloudWatchLogger():
    def __init__(self, is_prod):
        # Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        lg = log_group_name if is_prod else log_group_name+"_test"
        ls = log_stream_name if is_prod else log_stream_name+"_test"

        # Set up the CloudWatch log handler
        cloudwatch_handler = watchtower.CloudWatchLogHandler(
            log_group=lg,
            stream_name=ls,
            use_queues=True  # Use a background thread to send logs to CloudWatch
        )
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        cloudwatch_handler.setFormatter(formatter)
        
        # Add the CloudWatch handler to the logger
        self.logger.addHandler(cloudwatch_handler)
        
        # Optionally, add a console handler to also print logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)   
    
    def error(self, message):
        self.logger.error(message)
    
if __name__ == "__main__":
    logger = CloudWatchLogger() #logger initializes


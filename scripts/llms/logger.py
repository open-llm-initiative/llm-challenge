import logging
import watchtower

#Helper class to add cloud watch logs
class CloudWatchLogger():
    def __init__(self):
        # Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Set up the CloudWatch log handler
        cloudwatch_handler = watchtower.CloudWatchLogHandler(
            log_group="llm_challenge_log_group",
            stream_name="llm_challenge_log_stream",
            use_queues=True  # Use a background thread to send logs to CloudWatch
        )
        
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


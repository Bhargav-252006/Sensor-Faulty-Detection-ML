import os
import logging
from datetime import datetime

# 1. Define the log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 2. Define the directory where logs will be stored
# This will be 'your_current_working_directory/logs'
logs_dir = os.path.join(os.getcwd(), "logs")

# 3. Create the logs directory if it doesn't exist
# This will create 'your_current_working_directory/logs'
os.makedirs(logs_dir, exist_ok=True)

# 4. Define the full path to the log file
# This will be 'your_current_working_directory/logs/your_log_file_name.log'
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# 5. Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Optional: Add a test log message to confirm it works
# logging.info("Logging setup successful!")
# logging.warning("This is a warning message.")
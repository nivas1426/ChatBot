import re
import logging
from datetime import datetime

def maintain_conversation_context(context, user_input, max_context_size=5):
    """
    Update the conversation context with the new user input.
    
    :param context: List of previous conversation turns.
    :param user_input: New user input to add to the context.
    :param max_context_size: Maximum size of the context list.
    :return: Updated conversation context.
    """
    context.append(user_input)
    return context[-max_context_size:]

def preprocess_text(text):
    """
    Preprocess the input text by cleaning and normalizing it.
    
    :param text: Input text.
    :return: Cleaned and normalized text.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()  # Remove leading/trailing spaces

def setup_logging(log_file='app.log', log_level=logging.INFO):
    """
    Set up logging configuration.
    
    :param log_file: Path to the log file.
    :param log_level: Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            filename=log_file,
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        logging.getLogger().addHandler(logging.StreamHandler())

def log_message(message, level='info'):
    """
    Log a message at the specified log level.
    
    :param message: Message to log.
    :param level: Log level (e.g., 'info', 'error', 'warning', 'debug').
    """
    logger = logging.getLogger()
    log_levels = {
        'info': logging.INFO,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'debug': logging.DEBUG
    }
    logger.setLevel(log_levels.get(level, logging.INFO))
    log_function = {
        logging.INFO: logger.info,
        logging.ERROR: logger.error,
        logging.WARNING: logger.warning,
        logging.DEBUG: logger.debug
    }
    log_function.get(logger.level, logger.info)(message)

def format_response(results):
    """
    Format the search results into a response dictionary.
    
    :param results: List of search results.
    :return: Formatted response dictionary.
    """
    if not isinstance(results, list):
        raise ValueError("Results should be a list")
    response = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "results": results
    }
    return response


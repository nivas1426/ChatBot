from flask import Flask, request, jsonify, render_template
from model import get_embedding
from search import search
from utils import maintain_conversation_context
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

conversation_context = {}

# Casual responses
casual_responses = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! What can I do for you?",
    "hey": "Hey! How can I help?",
    "how are you": "I'm just a bot, but I'm here to help! How can I assist you?",
    "good morning": "Good morning! How can I assist you today?",
    "good afternoon": "Good afternoon! What can I do for you?",
    "good evening": "Good evening! How can I assist you?",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!"
}

# Keywords for price-related questions
price_keywords = ["cost", "price", "how much", "fee", "charge", "rate"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message').strip().lower()
        user_id = request.remote_addr  # Use IP as a simple user identifier

        logging.info(f"User {user_id} input: {user_input}")

        # Check for casual greetings
        if user_input in casual_responses:
            response = casual_responses[user_input]
            logging.info(f"Response to user {user_id}: {response}")
            return jsonify({'response': response})

        # Check for price-related questions
        if any(keyword in user_input for keyword in price_keywords):
            response = "Our wines vary in price depending on the type and vintage. For example, our Sauvignon Blanc is priced at $45 for non-members. For more specific information, please visit our website or contact us directly."
            logging.info(f"Response to user {user_id}: {response}")
            return jsonify({'response': response})

        conversation_context[user_id] = maintain_conversation_context(
            conversation_context.get(user_id, []), user_input
        )
        
        user_embedding = get_embedding(user_input)
        results = search(user_input)
        
        if results:
            response = results[0]['text']
        else:
            response = "I'm not sure about that. Please contact the business directly for more information."
        
        logging.info(f"Response to user {user_id}: {response}")
        return jsonify({'response': response})

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'response': "There was an error processing your request. Please try again later."})

if __name__ == '__main__':
    app.run(debug=True, port=5002)

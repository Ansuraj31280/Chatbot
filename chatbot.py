from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify, send_from_directory
import spacy
import json
from datetime import datetime
import os

# Load models
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
nlp = spacy.load("en_core_web_sm")

# Ensure JSON files exist and are properly formatted
def ensure_json_file(file_path, default_content):
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump(default_content, f)
    else:
        try:
            with open(file_path, "r") as f:
                json.load(f)
        except json.JSONDecodeError:
            with open(file_path, "w") as f:
                json.dump(default_content, f)

# Initialize JSON files
ensure_json_file("conversations.json", [])
ensure_json_file("feedback.json", [])
ensure_json_file("knowledge.json", {})

# Load knowledge base
with open("knowledge.json", "r") as f:
    knowledge = json.load(f)

def chat(input_text):
    # Check the knowledge base first
    if input_text.lower() in knowledge:
        return knowledge[input_text.lower()]

    # If not found, generate a response
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return output

def store_conversation(user_input, bot_response):
    conversation = {
        "user_input": user_input,
        "bot_response": bot_response,
        "timestamp": datetime.now().isoformat()
    }
    with open("conversations.json", "a") as f:
        f.write(json.dumps(conversation) + "\n")

def store_feedback(user_input, bot_response, feedback):
    feedback_entry = {
        "user_input": user_input,
        "bot_response": bot_response,
        "correction": feedback,
        "timestamp": datetime.now().isoformat()
    }
    with open("feedback.json", "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")

    # Update knowledge base
    knowledge[user_input.lower()] = feedback
    with open("knowledge.json", "w") as f:
        json.dump(knowledge, f)

app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

@app.route("/chat", methods=["POST"])
def chat_route():
    user_input = request.json.get("message")
    bot_response = chat(user_input)
    store_conversation(user_input, bot_response)
    return jsonify({"response": bot_response})

@app.route("/feedback", methods=["POST"])
def feedback_route():
    data = request.json
    user_input = data.get("user_input")
    bot_response = data.get("bot_response")
    feedback = data.get("feedback")
    store_feedback(user_input, bot_response, feedback)
    return jsonify({"response": "Feedback received and stored."})

if __name__ == "__main__":
    app.run(port=5000)

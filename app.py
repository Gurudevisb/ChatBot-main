from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from waitress import serve
import os  # Import os to fetch the PORT environment variable

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Check if CUDA is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

def get_Chat_response(text):
    # Initialize chat history
    chat_history_ids = None

    # Encode the new user input, add the eos_token, and return tensors in PyTorch
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt').to(device)

    # Generate the attention mask
    attention_mask = torch.ones(new_user_input_ids.shape, dtype=torch.long).to(device)

    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    # Update attention mask to include chat history
    if chat_history_ids is not None:
        attention_mask = torch.cat(
            [torch.ones(chat_history_ids.shape, dtype=torch.long).to(device), attention_mask],
            dim=-1
        )

    # Use torch.no_grad() to save memory during inference
    with torch.no_grad():
        # Generate a response while limiting the total chat history to 512 tokens
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=512,  # Reduced max_length to save memory
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask
        )

    # Clear CUDA cache after each response
    torch.cuda.empty_cache()

    # Return the last output tokens from the bot
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == "__main__":
    # Add a debugging print statement
    print("App is starting...")

    # Get the port from environment variables or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Use waitress to serve the Flask app
    serve(app, host="0.0.0.0", port=port)

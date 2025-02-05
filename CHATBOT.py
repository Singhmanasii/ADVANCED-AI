from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress Hugging Face symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def chatbot():
    # Load pre-trained GPT-2 model and tokenizer
    print("Loading model...")
    model_name = "gpt2"  # You can replace this with 'gpt2-medium' or another variant

    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set a padding token (GPT-2 doesn’t have one by default)
    tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!")

    # Instructions
    print("\nChatbot ready! Type 'exit' to quit.\n")

    # Chat loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Tokenize and generate a response with attention mask
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Generate the response
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,  # Include attention mask
            max_length=100,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id  # Use explicitly defined padding token
        )

        # Decode and print the response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Chatbot: {response[len(user_input):].strip()}")  # Remove user's input from response

# Run the chatbot
if __name__ == "__main__":
    chatbot()

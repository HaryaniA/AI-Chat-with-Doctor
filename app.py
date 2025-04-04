import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

class CustomChatDoctor:
    def __init__(self):
        print("Initializing ChatDoctor...")
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Model name
        model_name = "zl111/ChatDoctor"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
        
        # Define system prompt
        self.system_prompt = """You are a helpful, respectful and honest medical assistant. 
        Always answer as helpfully as possible, while being safe. 
        Your answers should be based on verified medical information.
        If a question doesn't make any sense, or is not factually coherent, explain why instead of answering something incorrect. 
        If you don't know the answer to a question, respond with "I don't have enough information to provide a reliable answer."
        """
        
        # Initialize conversation history
        self.conversation_history = []
    
    def generate_response(self, user_input):
        try:
            # Add user input to history
            self.conversation_history.append(f"User: {user_input}")
            
            # Construct the prompt
            prompt = self.system_prompt + "\n\n"
            prompt += "\n".join(self.conversation_history[-10:])
            prompt += "\nAI Assistant: "
            
            # Generate the response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Add response to history
            self.conversation_history.append(f"AI Assistant: {response}")
            
            return response
        
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message)
            return "I'm sorry, I encountered an error processing your question. Please try again."
    
    def reset_conversation(self):
        self.conversation_history = []
        return None

# Initialize the model
chat_doctor = CustomChatDoctor()

# Define example inputs
examples = [
    "What are the symptoms of diabetes?",
    "How can I manage my hypertension?",
    "What should I do for a persistent headache?"
]

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Your Custom ChatDoctor")
    gr.Markdown("Ask medical questions and get AI-powered responses.")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your medical question here...")
    
    with gr.Row():
        submit_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear Conversation")
    
    gr.Examples(examples=examples, inputs=msg)
    
    gr.Markdown("""
    ### Disclaimer
    This AI assistant provides information for educational purposes only. 
    Always consult with a qualified healthcare provider for medical advice.
    """)
    
    def respond(message, chat_history):
        bot_message = chat_doctor.generate_response(message)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    def clear_history():
        chat_doctor.reset_conversation()
        return None
    
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(clear_history, None, chatbot)

# Launch the app
demo.launch()
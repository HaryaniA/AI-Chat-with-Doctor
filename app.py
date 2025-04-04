import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gradio as gr
import os

# Setup cache directory for models
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'

class CustomChatDoctor:
    def __init__(self):
        print("Initializing ChatDoctor...")
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Model name
        model_name = "zl111/ChatDoctor"
        
        # Setup quantization for memory efficiency
        if self.device == "cuda":
            # Use 8-bit quantization if GPU is available
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # For CPU, use lighter settings
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        print("Model loaded successfully!")
        
        # Define system prompt
        self.system_prompt = """You are a helpful, respectful and honest medical assistant. 
        Always answer as helpfully as possible, while being safe. 
        Your answers should be based on verified medical information.
        If a question doesn't make any sense, or is not factually coherent, explain why instead of answering something incorrect. 
        If you don't know the answer to a question, respond with "I don't have enough information to provide a reliable answer."
        Always include a disclaimer that you are an AI assistant and not a licensed medical professional.
        """
        
        # Initialize conversation history
        self.reset_conversation()
    
    def generate_response(self, user_input):
        try:
            # Add user input to history
            self.conversation_history.append(f"User: {user_input}")
            
            # Construct the prompt
            prompt = self.system_prompt + "\n\n"
            prompt += "\n".join(self.conversation_history[-10:])
            prompt += "\nAI Assistant: "
            
            # Generate the response with appropriate parameters based on device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():  # Disable gradient calculations for inference
                generation_config = {
                    "max_new_tokens": 512,
                    "temperature": 0.7, 
                    "top_p": 0.9,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.eos_token_id
                }
                
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generation_config
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Add response to history
            self.conversation_history.append(f"AI Assistant: {response}")
            
            return response
        
        except Exception as e:
            # Handle any errors during generation
            error_message = f"An error occurred: {str(e)}"
            print(error_message)
            return "I'm sorry, I encountered an error processing your question. Please try again or ask a different question."
    
    def reset_conversation(self):
        self.conversation_history = []
        return None

# Create a singleton instance to avoid reloading the model for each user
chat_doctor = None

def get_chat_doctor():
    global chat_doctor
    if chat_doctor is None:
        chat_doctor = CustomChatDoctor()
    return chat_doctor

# Example inputs to help users get started
examples = [
    "What are the symptoms of diabetes?",
    "How can I manage my hypertension?",
    "What should I do for a persistent headache?",
    "Can you explain what asthma is?",
    "What are the side effects of ibuprofen?"
]

# Add CSS for styling
css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.disclaimer {
    margin-top: 20px;
    padding: 10px;
    background-color: #f8f9fa;
    border-left: 3px solid #f0ad4e;
    font-size: 14px;
}
"""

# Create welcome message function
def welcome():
    return "Welcome to ChatDoctor! I'm an AI assistant trained to provide medical information. How can I help you today?"

# Build the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# Your Custom ChatDoctor")
    gr.Markdown("Ask medical questions and get AI-powered responses.")
    
    chatbot = gr.Chatbot(height=600, type="messages")
    msg = gr.Textbox(placeholder="Type your medical question here...", lines=2)
    
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear Conversation")
    
    # Display example queries that users can click on
    gr.Examples(
        examples=examples,
        inputs=msg
    )
    
    with gr.Accordion("About this AI", open=False):
        gr.Markdown("""
        **ChatDoctor** is a medical conversation model designed to provide general health information.
        
        This AI uses language models to generate responses based on patterns learned from medical texts and conversations.
        
        **Important Notes:**
        - This system is for informational purposes only
        - Not a substitute for professional medical advice
        - In emergencies, contact emergency services immediately
        """)
    
    # Add disclaimer at the bottom with custom styling
    gr.HTML("""
    <div class="disclaimer">
        <strong>Disclaimer:</strong> This AI assistant provides information for educational purposes only. 
        Always consult with a qualified healthcare provider for medical advice, diagnosis, or treatment. 
        This tool is not intended to replace professional medical consultation.
    </div>
    """)
    
    def respond(message, chat_history):
        # Lazy-load model on first request
        doctor = get_chat_doctor()
        
        if message.strip() == "":
            return "", chat_history
            
        bot_message = doctor.generate_response(message)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history
    
    def clear_history():
        doctor = get_chat_doctor()
        doctor.reset_conversation()
        return None
    
    # Show welcome message when the app starts
    demo.load(lambda: None, None, chatbot, js="""
        () => {
            const welcomeMsg = "Welcome to ChatDoctor! I'm an AI assistant trained to provide medical information. How can I help you today?";
            return [[null, welcomeMsg]];
        }
    """)
    
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(clear_history, None, chatbot)

# Launch the app
demo.launch()
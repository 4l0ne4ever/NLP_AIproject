import gradio as gr
import os
import json
import boto3
from datetime import datetime
from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognizer
from character_network import CharacterNetworkGenerator
from text_classification import LocationClassifier
from character_chatbot import CharacterChatbot
from character_chatbot.character_chatbotQwen import CharacterChatbotQwen
from dotenv import load_dotenv
from config import (
    S3_BUCKET_NAME, S3_REGION, S3_MODEL_PATHS, LOCAL_MODEL_CACHE,
    FALLBACK_MODELS, DEFAULT_THEMES, GRADIO_CONFIG, FALLBACK_CONFIG
)

load_dotenv()

class S3ModelManager:
    """Manages loading and downloading models from S3"""
    
    def __init__(self, bucket_name=S3_BUCKET_NAME):
        self.s3_client = boto3.client('s3', region_name=S3_REGION)
        self.bucket_name = bucket_name
        self.local_model_cache = LOCAL_MODEL_CACHE
        os.makedirs(self.local_model_cache, exist_ok=True)
    
    def get_latest_model_info(self, model_type):
        """Get the latest trained model information from S3"""
        try:
            latest_key = f"models/trained/{model_type}/latest.json"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=latest_key)
            return json.loads(response['Body'].read())
        except Exception as e:
            print(f"Error getting latest model info for {model_type}: {e}")
            return None
    
    def download_model(self, model_path, local_path):
        """Download model from S3 to local cache"""
        try:
            if not os.path.exists(local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                self.s3_client.download_file(self.bucket_name, model_path, local_path)
            return local_path
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None
    
    def get_model_path(self, model_type):
        """Get local path for the latest model"""
        model_info = self.get_latest_model_info(model_type)
        if model_info:
            model_s3_path = model_info.get('model_path')
            if model_s3_path:
                local_path = os.path.join(self.local_model_cache, model_type, 
                                        model_info.get('timestamp', 'latest'))
                return self.download_model(model_s3_path, local_path)
        return None

# Initialize S3 Model Manager
s3_manager = S3ModelManager()

# Global model instances - will be loaded on demand
character_chatbot_llama = None
character_chatbot_qwen = None
fallback_messages = []  # Store fallback messages for display

def announce_fallback(model_type):
    """Announce when falling back to HuggingFace model"""
    global fallback_messages
    if FALLBACK_CONFIG.get("announce_fallback", True):
        message = FALLBACK_CONFIG.get("fallback_message", 
                                    "Notice: Using HuggingFace fallback model for {model_type}. Custom trained model not available.").format(model_type=model_type)
        print(message)
        fallback_messages.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": model_type,
            "message": message
        })
        return message
    return None

def initialize_chatbot_models():
    """Initialize chatbot models from S3 or fallback to HuggingFace"""
    global character_chatbot_llama, character_chatbot_qwen
    
    # Try to load LLama model from S3
    llama_model_path = s3_manager.get_model_path("llama")
    if llama_model_path:
        print(f"Loading LLama model from S3: {llama_model_path}")
        character_chatbot_llama = CharacterChatbot(
            llama_model_path,
            huggingface_token=os.getenv('huggingface_token'),
        )
    else:
        # Fallback to HuggingFace model
        announce_fallback("llama")
        character_chatbot_llama = CharacterChatbot(
            FALLBACK_MODELS["llama"],
            huggingface_token=os.getenv('huggingface_token'),
        )
    
    # Try to load Qwen model from S3
    qwen_model_path = s3_manager.get_model_path("qwen")
    if qwen_model_path:
        print(f"Loading Qwen model from S3: {qwen_model_path}")
        character_chatbot_qwen = CharacterChatbotQwen(
            qwen_model_path,
            huggingface_token=os.getenv('huggingface_token'),
        )
    else:
        # Fallback to HuggingFace model
        announce_fallback("qwen")
        character_chatbot_qwen = CharacterChatbotQwen(
            FALLBACK_MODELS["qwen"],
            huggingface_token=os.getenv('huggingface_token'),
        )

def get_model_status():
    """Get current model status and training information"""
    status_info = []
    
    # Check LLama model
    llama_info = s3_manager.get_latest_model_info("llama")
    if llama_info:
        status_info.append(f"LLama Model: Trained on {llama_info.get('timestamp', 'Unknown')} - Accuracy: {llama_info.get('accuracy', 'N/A')}")
    else:
        status_info.append("LLama Model: Using HuggingFace fallback")
    
    # Check Qwen model
    qwen_info = s3_manager.get_latest_model_info("qwen")
    if qwen_info:
        status_info.append(f"Qwen Model: Trained on {qwen_info.get('timestamp', 'Unknown')} - Accuracy: {qwen_info.get('accuracy', 'N/A')}")
    else:
        status_info.append("Qwen Model: Using HuggingFace fallback")
    
    # Add fallback announcement setting status
    announcement_status = "ON" if FALLBACK_CONFIG.get("announce_fallback", True) else "OFF"
    status_info.append(f"\nFallback Announcements: {announcement_status}")
    
    # Add recent fallback messages if any
    global fallback_messages
    if fallback_messages:
        status_info.append("\nRecent Fallback Messages:")
        for msg in fallback_messages[-3:]:  # Show last 3 messages
            status_info.append(f"  [{msg['timestamp']}] {msg['message']}")
    
    return "\n".join(status_info)

def character_chatbot_withQwen(message, history):
    """Chat with Qwen character chatbot"""
    global character_chatbot_qwen
    if character_chatbot_qwen is None:
        initialize_chatbot_models()
    
    try:
        output = character_chatbot_qwen.chat(message, history)
        return output.strip()
    except Exception as e:
        return f"Error in Qwen chatbot: {str(e)}"

def chat_with_character_llama(message, history):
    """Chat with LLama character chatbot"""
    global character_chatbot_llama
    if character_chatbot_llama is None:
        initialize_chatbot_models()
    
    try:
        output = character_chatbot_llama.chat(message, history)
        if isinstance(output, dict) and 'content' in output:
            return output['content'].strip()
        return str(output).strip()
    except Exception as e:
        return f"Error in LLama chatbot: {str(e)}"

def get_themes(theme_list_str, subtitles_path, save_path):
    """Extract themes from subtitles using theme classifier"""
    try:
        if not theme_list_str or not subtitles_path:
            return "Please provide both theme list and subtitles path"
        
        theme_list = [theme.strip() for theme in theme_list_str.split(",")]
        theme_classifier = ThemeClassifier(theme_list)
        output_df = theme_classifier.get_themes(subtitles_path, save_path)
        
        output_df = output_df[theme_list]
        theme_scores = output_df.sum().reset_index()
        theme_scores.columns = ["Theme", "Score"]
        
        output_chart = gr.BarPlot(
            theme_scores,
            x="Theme",
            y="Score",
            title="Series Themes Analysis",
            tooltip=["Theme", "Score"],
            vertical=False,
            width=500,
            height=260,
        )
        return output_chart
    except Exception as e:
        return f"Error in theme analysis: {str(e)}"

def get_character_network(subtitles_path, ner_path):
    """Generate character network from subtitles"""
    try:
        if not subtitles_path:
            return "Please provide subtitles path", "Error: No subtitles path provided"
        
        ner = NamedEntityRecognizer()
        ner_df = ner.get_ners(subtitles_path, ner_path)
        
        character_network_generator = CharacterNetworkGenerator()
        relations_df = character_network_generator.generate_character_network(ner_df)
        html = character_network_generator.draw_network_graph(relations_df)
        return html, "Character network generated successfully"
    except Exception as e:
        return f"Error generating character network: {str(e)}", f"Failed: {str(e)}"
    
def classify_text(text_classification_model, text_classification_data_path, text_to_classify):
    """Classify text using location classifier"""
    try:
        if not text_to_classify:
            return "Please provide text to classify"
        
        location_classifier = LocationClassifier(
            model_path=text_classification_model,
            data_path=text_classification_data_path,
            huggingface_token=os.getenv('huggingface_token')
        )
        output = location_classifier.classify_location(text_to_classify)
        return output
    except Exception as e:
        return f"Error in text classification: {str(e)}"

def refresh_models():
    """Refresh models from S3"""
    global character_chatbot_llama, character_chatbot_qwen
    character_chatbot_llama = None
    character_chatbot_qwen = None
    initialize_chatbot_models()
    return get_model_status()

def main():
    """Main Gradio interface"""
    
    # Custom CSS for clean styling without icons
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .section {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: #f8f9fa;
    }
    .section h2 {
        color: #333;
        border-bottom: 2px solid #667eea;
        padding-bottom: 10px;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Stranger Things AI Analysis Suite") as interface:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>Stranger Things AI Analysis Suite</h1>
            <p>Advanced NLP tools for theme analysis, character networks, and interactive chatbots</p>
        </div>
        """)
        
        # Model Status Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<div class='section'><h2>Model Status</h2></div>")
                model_status = gr.Textbox(
                    label="Current Model Status", 
                    value=get_model_status(),
                    interactive=False,
                    lines=4
                )
                refresh_button = gr.Button("Refresh Models from S3", variant="primary")
                refresh_button.click(refresh_models, outputs=[model_status])
        
        # Theme Classification Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<div class='section'><h2>Theme Classification</h2></div>")
                gr.Markdown("Analyze themes in subtitles using zero-shot classification")
                
                with gr.Row():
                    with gr.Column():
                        theme_plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(
                            label="Themes (comma-separated)", 
                            placeholder=", ".join(DEFAULT_THEMES[:4]),
                            value=", ".join(DEFAULT_THEMES[:4])
                        )
                        subtitles_path_themes = gr.Textbox(
                            label="Subtitles Path", 
                            placeholder="/path/to/subtitles.srt"
                        )
                        save_path_themes = gr.Textbox(
                            label="Save Path", 
                            placeholder="/path/to/save/results.csv"
                        )
                        get_themes_button = gr.Button("Analyze Themes", variant="primary")
                        get_themes_button.click(
                            get_themes, 
                            inputs=[theme_list, subtitles_path_themes, save_path_themes], 
                            outputs=[theme_plot]
                        )
        
        # Character Network Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<div class='section'><h2>Character Network Analysis</h2></div>")
                gr.Markdown("Generate interactive character relationship networks")
                
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                        network_status = gr.Textbox(label="Status", interactive=False)
                    with gr.Column():
                        subtitles_path_network = gr.Textbox(
                            label="Subtitles Path", 
                            placeholder="/path/to/subtitles.srt"
                        )
                        ner_path = gr.Textbox(
                            label="NER Save Path", 
                            placeholder="/path/to/save/ner_results.csv"
                        )
                        get_network_button = gr.Button("Generate Network", variant="primary")
                        get_network_button.click(
                            get_character_network, 
                            inputs=[subtitles_path_network, ner_path], 
                            outputs=[network_html, network_status]
                        )
        
        # Text Classification Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<div class='section'><h2>Location Classification</h2></div>")
                gr.Markdown("Classify text locations using fine-tuned language models")
                
                with gr.Row():
                    with gr.Column():
                        classification_output = gr.Textbox(
                            label="Classification Result", 
                            interactive=False,
                            lines=3
                        )
                    with gr.Column():
                        model_path = gr.Textbox(
                            label="Model Path", 
                            placeholder="/path/to/model or huggingface/model"
                        )
                        data_path = gr.Textbox(
                            label="Data Path", 
                            placeholder="/path/to/training/data.csv"
                        )
                        text_to_classify = gr.Textbox(
                            label="Text to Classify", 
                            placeholder="Enter text here...",
                            lines=3
                        )
                        classify_button = gr.Button("Classify Location", variant="primary")
                        classify_button.click(
                            classify_text, 
                            inputs=[model_path, data_path, text_to_classify], 
                            outputs=[classification_output]
                        )
        
        # Character Chatbot LLama Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<div class='section'><h2>Character Chatbot - LLama Model</h2></div>")
                gr.Markdown("Chat with Stranger Things characters using LLama-based models")
                
                llama_chatbot = gr.ChatInterface(
                    chat_with_character_llama,
                    chatbot=gr.Chatbot(height=400),
                    textbox=gr.Textbox(placeholder="Ask me about Stranger Things characters...", scale=7),
                    submit_btn="Send"
                )
        
        # Character Chatbot Qwen Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<div class='section'><h2>Character Chatbot - Qwen Model</h2></div>")
                gr.Markdown("Chat with Stranger Things characters using Qwen-based models")
                
                qwen_chatbot = gr.ChatInterface(
                    character_chatbot_withQwen,
                    chatbot=gr.Chatbot(height=400),
                    textbox=gr.Textbox(placeholder="Talk to Stranger Things characters...", scale=7),
                    submit_btn="Send"
                )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666; border-top: 1px solid #e0e0e0; margin-top: 30px;">
            <p>Stranger Things AI Analysis Suite - Powered by Custom Trained Models</p>
            <p>Models automatically loaded from S3 when available, with HuggingFace fallbacks</p>
        </div>
        """)
    
    return interface

if __name__ == "__main__":
    print("Initializing Stranger Things AI Analysis Suite...")
    print("Loading models from S3...")
    
    # Initialize models on startup
    try:
        initialize_chatbot_models()
        print("Models initialized successfully!")
    except Exception as e:
        print(f"Warning: Error initializing models: {e}")
        print("Will attempt to initialize on first use.")
    
    interface = main()
    interface.launch(**GRADIO_CONFIG)

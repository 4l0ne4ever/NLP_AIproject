import gradio as gr
import os
from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognizer
from character_network import CharacterNetworkGenerator
from text_classification import LocationClassifier
from character_chatbot import CharacterChatbot
from character_chatbot.character_chatbotQwen import CharacterChatbotQwen
from dotenv import load_dotenv

load_dotenv()
chatbot_qwen = CharacterChatbotQwen(
    "christopherxzyx/StrangerThings_Qwen-3-4B",
    huggingface_token=os.getenv('huggingface_token'),
)
def character_chatbot_withQwen(message, history):
    output = chatbot_qwen.chat(message, history)
    return output.strip()
def get_themes(theme_list_str, subtitles_path, save_path):
    try:
        theme_list = theme_list_str.split(",")
        theme_classifier = ThemeClassifier(theme_list)
        output_df = theme_classifier.get_themes(subtitles_path, save_path)
        
        output_df = output_df[theme_list]
        output_df = output_df[theme_list].sum().reset_index()
        output_df.columns = ["Theme", "Score"]
        
        output_chart = gr.BarPlot(
            output_df,
            x="Theme",
            y="Score",
            title="Series Themes",
            tooltip=["Theme", "Score"],
            vertical=False,
            width=500,
            height=260,
        )
        return output_chart
    except Exception as e:
        return f"Error: {str(e)}"

def get_character_network(subtitles_path, ner_path):
    try:
        ner = NamedEntityRecognizer()
        ner_df = ner.get_ners(subtitles_path, ner_path)
        
        character_network_generator = CharacterNetworkGenerator()
        relations_df = character_network_generator.generate_character_network(ner_df)
        html = character_network_generator.draw_network_graph(relations_df)
        return html, "Network generated successfully"
    except Exception as e:
        return f"Error: {str(e)}", f"Failed: {str(e)}"
    
def classify_text(text_classification_model, text_classification_data_path, text_to_classify):
    try:
        location_classifier = LocationClassifier(model_path=text_classification_model,
                                                data_path=text_classification_data_path,
                                                huggingface_token=os.getenv('huggingface_token'))
        output = location_classifier.classify_location(text_to_classify)
        return output
    except Exception as e:
        return f"Error: {str(e)}"

def chat_with_character(message, history):
    character_chatbot = CharacterChatbot(
        "christopherxzyx/StrangerThings_Llama-3-8B_v3",
        huggingface_token=os.getenv('huggingface_token'),
    )
    output = character_chatbot.chat(message,history)
    output = output['content'].strip()
    return output
    
def main():
    with gr.Blocks() as interface:
        # Theme Classification (Zero Shot Classifiers)
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Classifiers)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes")
                        subtitles_path = gr.Textbox(label="Subtitles or script path")
                        save_path = gr.Textbox(label="Save path")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs=[theme_list, subtitles_path, save_path], outputs=[plot])
        # Character Network (NERs and Graphs)
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                        status = gr.Textbox(label="Status")  # Thêm ô trạng thái
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Subtitles or script path")
                        ner_path = gr.Textbox(label="NERs save path")
                        get_network_graph_button = gr.Button("Get Network")
                        get_network_graph_button.click(get_character_network, inputs=[subtitles_path, ner_path], outputs=[network_html, status])
        # Text Classification (LLMs)
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Text Classification (LLMs)</h1>")
                with gr.Row():
                    with gr.Column():
                        text_classification_output = gr.Textbox(label="Text Classification Output")
                    with gr.Column():
                        text_classification_model = gr.Textbox(label="Model path")
                        text_classification_data_path = gr.Textbox(label="Data path")
                        text_to_classify = gr.Textbox(label="Text to classify")
                        classify_text_button = gr.Button("Classify Text (Location)")
                        classify_text_button.click(classify_text, inputs=[text_classification_model, text_classification_data_path, text_to_classify], outputs=[text_classification_output])
                        
        # Character Chatbot (LLMs)
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Chatbot (LLMs)</h1>")
                gr.ChatInterface(chat_with_character)
        
        #Character Chatbot Qwen 
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Chatbot (Qwen)</h1>")
                gr.ChatInterface(character_chatbot_withQwen)
                
    interface.launch(share=True, debug=True)  # Bật debug để xem log

if __name__ == "__main__":
    main()
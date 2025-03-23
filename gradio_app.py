import gradio as gr
from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognizer
from character_network import CharacterNetworkGenerator
def get_themes(theme_list_str, subtitles_path, save_path):
    theme_list = theme_list_str.split(",")
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path,save_path)
    
    output_df = output_df[theme_list]
    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ["Theme", "Score"]
    
    output_chart = gr.BarPlot(
        output_df,
        x="Theme",
        y="Score",
        title="Series Themes",
        tooltip=["Theme", "Score"],
        vertical = False,
        width=500,
        height=260,
    )
    return output_chart
def get_character_network(subtitles_path, ner_path):
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path, ner_path)
    
    character_network_generator = CharacterNetworkGenerator()
    relations_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relations_df)
    return html
    
        
def main():
    with gr.Blocks() as interface:
        #Theme Classification Session
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Classifiers)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label = "Themes")
                        subtitles_path = gr.Textbox(label = "Subtitles or script path")
                        save_path = gr.Textbox(label = "Save path")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs = [theme_list, subtitles_path, save_path], outputs = [plot])
        #Character Network Session                
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label = "Subtitles or script path")
                        ner_path = gr.Textbox(label = "NERs save path")
                        get_network_graph_button = gr.Button("Get Network")
                        get_network_graph_button.click(get_character_network, inputs = [subtitles_path,ner_path], outputs = [network_html])
    interface.launch(share=True)
if __name__ == "__main__":
    main()   
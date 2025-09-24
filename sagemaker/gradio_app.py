"""
SageMaker-Enabled Gradio App for Stranger Things NLP Project

This is a modified version of the main Gradio app that uses SageMaker endpoints
for inference instead of loading models locally. This provides better scalability,
cost efficiency, and production readiness.
"""

import gradio as gr
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

# Import our SageMaker modules
from config import SageMakerConfigManager
from deployment_manager import SageMakerDeploymentManager
from monitoring import SageMakerMonitor

# Import original modules for components that don't use SageMaker endpoints yet
import sys
sys.path.append(str(Path(__file__).parent.parent))

from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognizer, CharacterNetworkGenerator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerGradioApp:
    """SageMaker-powered Gradio application"""
    
    def __init__(self, config_file: str = None):
        # Initialize SageMaker components
        self.config = SageMakerConfigManager(config_file)
        self.deployment_manager = SageMakerDeploymentManager(self.config)
        self.s3_manager = SageMakerS3Manager(
            bucket_name=self.config.s3_config.bucket_name,
            region=self.config.s3_config.region
        )
        
        # Track active endpoints for different models
        self.endpoints = {
            'chatbot_llama': None,
            'chatbot_qwen': None,
            'text_classifier': None
        }
        
        # Auto-discover endpoints on startup
        self._discover_endpoints()
    
    def _discover_endpoints(self):
        """Discover and connect to existing SageMaker endpoints"""
        try:
            active_endpoints = self.deployment_manager.list_active_endpoints()
            logger.info(f"🔍 Found {len(active_endpoints)} active endpoints")
            
            for endpoint in active_endpoints:
                name = endpoint['name']
                status = endpoint['status']
                
                if status == 'InService':
                    # Match endpoints to model types based on naming conventions
                    if 'llama' in name.lower() or 'chatbot' in name.lower():
                        self.endpoints['chatbot_llama'] = name
                        logger.info(f"✅ Connected to Llama chatbot endpoint: {name}")
                    elif 'qwen' in name.lower():
                        self.endpoints['chatbot_qwen'] = name
                        logger.info(f"✅ Connected to Qwen chatbot endpoint: {name}")
                    elif 'classifier' in name.lower() or 'location' in name.lower():
                        self.endpoints['text_classifier'] = name
                        logger.info(f"✅ Connected to text classifier endpoint: {name}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not discover endpoints: {e}")
    
    def invoke_chatbot_endpoint(self, message: str, history: List, model_type: str = 'llama') -> str:
        """Invoke chatbot endpoint for inference"""
        endpoint_key = f'chatbot_{model_type}'
        endpoint_name = self.endpoints.get(endpoint_key)
        
        if not endpoint_name:
            return f"❌ No {model_type} chatbot endpoint available. Please deploy a model first."
        
        try:
            # Prepare the conversation context
            conversation_context = []
            
            # Add system message for Eleven
            conversation_context.append({
                "role": "system",
                "content": "You are Eleven (or El for short), a character from the Netflix series Stranger Things. Your responses should reflect her personality and speech patterns."
            })
            
            # Add conversation history
            for user_msg, assistant_msg in history:
                conversation_context.append({"role": "user", "content": user_msg})
                conversation_context.append({"role": "assistant", "content": assistant_msg})
            
            # Add current message
            conversation_context.append({"role": "user", "content": message})
            
            # Prepare payload for SageMaker endpoint
            payload = {
                "inputs": message,
                "conversation_history": conversation_context[-10:],  # Keep last 10 exchanges
                "parameters": {
                    "max_length": 256,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_p": 0.9,
                    "pad_token_id": 128001  # Typical for Llama models
                }
            }
            
            # Invoke endpoint
            response = self.deployment_manager.invoke_endpoint(endpoint_name, payload)
            
            # Extract response text
            if isinstance(response, dict):
                return response.get('generated_text', response.get('outputs', str(response)))
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"❌ Chatbot inference error: {e}")
            return f"❌ Sorry, I'm having trouble responding right now: {str(e)}"
    
    def invoke_text_classifier_endpoint(self, text: str) -> str:
        """Invoke text classification endpoint"""
        endpoint_name = self.endpoints.get('text_classifier')
        
        if not endpoint_name:
            return "❌ No text classifier endpoint available. Please deploy a model first."
        
        try:
            payload = {
                "inputs": text,
                "parameters": {
                    "return_all_scores": True
                }
            }
            
            response = self.deployment_manager.invoke_endpoint(endpoint_name, payload)
            
            # Format response
            if isinstance(response, dict) and 'predictions' in response:
                predictions = response['predictions']
                result = f"🎯 Classification Results:\\n"
                for pred in predictions:
                    result += f"  • {pred['label']}: {pred['score']:.3f}\\n"
                return result
            else:
                return f"📍 Classification: {response}"
                
        except Exception as e:
            logger.error(f"❌ Text classifier error: {e}")
            return f"❌ Classification failed: {str(e)}"
    
    def get_endpoint_status(self) -> str:
        """Get status of all endpoints"""
        status_info = []
        status_info.append("🚀 **SageMaker Endpoint Status**\\n")
        
        for model_type, endpoint_name in self.endpoints.items():
            if endpoint_name:
                try:
                    status = self.deployment_manager.get_endpoint_status(endpoint_name)
                    emoji = "✅" if status['status'] == 'InService' else "⚠️"
                    status_info.append(f"{emoji} **{model_type.title()}**: {status['status']}")
                    status_info.append(f"   └─ Endpoint: `{endpoint_name}`")
                except Exception as e:
                    status_info.append(f"❌ **{model_type.title()}**: Error - {str(e)}")
            else:
                status_info.append(f"⭕ **{model_type.title()}**: No endpoint deployed")
        
        # Add deployment summary
        try:
            summary = self.deployment_manager.get_deployment_summary()
            status_info.append(f"\\n📊 **Summary**: {summary['endpoints']} endpoints, {summary['models']} models")
        except Exception:
            pass
        
        return "\\n".join(status_info)
    
    def deploy_model_interface(self, model_artifacts_uri: str, model_type: str) -> str:
        """Interface for deploying models from Gradio"""
        if not model_artifacts_uri.startswith('s3://'):
            return "❌ Please provide a valid S3 URI for model artifacts"
        
        try:
            # Generate model name
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_name = f"stranger-things-{model_type}-{timestamp}"
            
            # Deploy model
            deployment_info = self.deployment_manager.deploy_model_complete(
                model_name=model_name,
                model_artifacts_s3_uri=model_artifacts_uri
            )
            
            # Update endpoint tracking
            endpoint_name = deployment_info['endpoint_name']
            if model_type in ['llama', 'qwen']:
                self.endpoints[f'chatbot_{model_type}'] = endpoint_name
            elif model_type == 'text_classifier':
                self.endpoints['text_classifier'] = endpoint_name
            
            result = f"🚀 **Deployment Initiated Successfully!**\\n\\n"
            result += f"📦 **Model**: {model_name}\\n"
            result += f"🔗 **Endpoint**: {endpoint_name}\\n"
            result += f"⏳ **Status**: {deployment_info['status']}\\n\\n"
            result += f"💡 Endpoint creation takes 5-10 minutes. Check status tab for updates."
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return f"❌ **Deployment Failed**: {str(e)}"
    
    def create_gradio_interface(self):
        """Create the main Gradio interface"""
        
        # Custom CSS for better SageMaker branding
        custom_css = """
        .sagemaker-header {
            background: linear-gradient(90deg, #232F3E 0%, #FF9900 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 20px;
        }
        .endpoint-status {
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
        }
        .tab-nav {
            background-color: #232F3E !important;
            color: white !important;
        }
        """
        
        with gr.Blocks(css=custom_css, title="Stranger Things NLP - SageMaker Edition") as app:
            
            # Header
            gr.HTML("""
            <div class="sagemaker-header">
                <h1>🎬 Stranger Things NLP - SageMaker Edition</h1>
                <p>Powered by AWS SageMaker for scalable, production-ready inference</p>
            </div>
            """)
            
            with gr.Tabs():
                
                # SageMaker Status Tab
                with gr.Tab("🚀 SageMaker Status"):
                    gr.Markdown("## Endpoint Status & Management")
                    
                    with gr.Row():
                        with gr.Column():
                            status_output = gr.Markdown(
                                value=self.get_endpoint_status(),
                                elem_classes=["endpoint-status"]
                            )
                            refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                            refresh_btn.click(
                                fn=self.get_endpoint_status,
                                outputs=[status_output]
                            )
                        
                        with gr.Column():
                            gr.Markdown("### Quick Deploy Model")
                            model_artifacts_input = gr.Textbox(
                                label="Model Artifacts S3 URI",
                                placeholder="s3://bucket-name/models/model.tar.gz"
                            )
                            model_type_dropdown = gr.Dropdown(
                                choices=["llama", "qwen", "text_classifier"],
                                label="Model Type",
                                value="llama"
                            )
                            deploy_btn = gr.Button("🚀 Deploy Model", variant="primary")
                            deploy_output = gr.Markdown()
                            
                            deploy_btn.click(
                                fn=self.deploy_model_interface,
                                inputs=[model_artifacts_input, model_type_dropdown],
                                outputs=[deploy_output]
                            )
                
                # Character Chatbot Tabs
                with gr.Tab("🤖 Character Chatbot (Llama)"):
                    gr.Markdown("## Chat with Eleven - Llama Model")
                    gr.Markdown("*Powered by SageMaker real-time inference endpoint*")
                    
                    def chat_llama(message, history):
                        return self.invoke_chatbot_endpoint(message, history, 'llama')
                    
                    gr.ChatInterface(
                        fn=chat_llama,
                        title="Eleven (Llama)",
                        description="Chat with Eleven using the Llama-based model"
                    )
                
                with gr.Tab("🧠 Character Chatbot (Qwen)"):
                    gr.Markdown("## Chat with Eleven - Qwen Model")
                    gr.Markdown("*Powered by SageMaker real-time inference endpoint*")
                    
                    def chat_qwen(message, history):
                        return self.invoke_chatbot_endpoint(message, history, 'qwen')
                    
                    gr.ChatInterface(
                        fn=chat_qwen,
                        title="Eleven (Qwen)",
                        description="Chat with Eleven using the Qwen-based model"
                    )
                
                # Text Classification Tab
                with gr.Tab("📍 Text Classification (SageMaker)"):
                    gr.Markdown("## Location Classification")
                    gr.Markdown("*Powered by SageMaker endpoint*")
                    
                    with gr.Row():
                        with gr.Column():
                            text_input = gr.Textbox(
                                label="Text to Classify",
                                placeholder="Enter text to classify its location...",
                                lines=3
                            )
                            classify_btn = gr.Button("🎯 Classify Text", variant="primary")
                        
                        with gr.Column():
                            classification_output = gr.Markdown(label="Classification Results")
                    
                    classify_btn.click(
                        fn=self.invoke_text_classifier_endpoint,
                        inputs=[text_input],
                        outputs=[classification_output]
                    )
                
                # Theme Classification Tab (Still uses local processing)
                with gr.Tab("🎭 Theme Classification"):
                    gr.Markdown("## Theme Classification (Zero-Shot)")
                    gr.Markdown("*Note: This still uses local processing - will be migrated to SageMaker in future versions*")
                    
                    with gr.Row():
                        with gr.Column():
                            plot_output = gr.BarPlot()
                        with gr.Column():
                            theme_list = gr.Textbox(label="Themes (comma-separated)")
                            subtitles_path = gr.Textbox(label="Subtitles or script path")
                            save_path = gr.Textbox(label="Save path")
                            get_themes_button = gr.Button("🎭 Get Themes")
                            
                            def get_themes_local(theme_list_str, subtitles_path, save_path):
                                try:
                                    # Upload data to S3 first for better integration
                                    if subtitles_path and os.path.exists(subtitles_path):
                                        s3_data_path = self.s3_manager.upload_directory(
                                            subtitles_path, 
                                            "theme-analysis/input/"
                                        )
                                        logger.info(f"📤 Uploaded theme data to S3: {s3_data_path}")
                                    
                                    # Use original theme classifier
                                    theme_list = theme_list_str.split(",")
                                    theme_classifier = ThemeClassifier(theme_list)
                                    output_df = theme_classifier.get_themes(subtitles_path, save_path)
                                    
                                    output_df = output_df[theme_list]
                                    output_df = output_df[theme_list].sum().reset_index()
                                    output_df.columns = ["Theme", "Score"]
                                    
                                    # Save results to S3
                                    results_s3_path = f"theme-analysis/results/themes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                    temp_path = "/tmp/theme_results.csv"
                                    output_df.to_csv(temp_path, index=False)
                                    self.s3_manager.upload_file(temp_path, results_s3_path)
                                    logger.info(f"💾 Saved theme results to S3: {results_s3_path}")
                                    
                                    return gr.BarPlot(
                                        output_df,
                                        x="Theme",
                                        y="Score",
                                        title="Series Themes",
                                        tooltip=["Theme", "Score"],
                                        vertical=False,
                                        width=500,
                                        height=260,
                                    )
                                except Exception as e:
                                    logger.error(f"❌ Theme analysis error: {e}")
                                    return gr.BarPlot()
                            
                            get_themes_button.click(
                                get_themes_local,
                                inputs=[theme_list, subtitles_path, save_path],
                                outputs=[plot_output]
                            )
                
                # Character Network Tab (Still uses local processing)
                with gr.Tab("🕸️ Character Network"):
                    gr.Markdown("## Character Network Analysis")
                    gr.Markdown("*Note: This still uses local processing - will be migrated to SageMaker in future versions*")
                    
                    with gr.Row():
                        with gr.Column():
                            network_html = gr.HTML()
                            status_text = gr.Textbox(label="Status", interactive=False)
                        with gr.Column():
                            subtitles_path_network = gr.Textbox(label="Subtitles or script path")
                            ner_path = gr.Textbox(label="NERs save path")
                            get_network_button = gr.Button("🕸️ Generate Network")
                            
                            def get_character_network_local(subtitles_path, ner_path):
                                try:
                                    # Upload data to S3 first
                                    if subtitles_path and os.path.exists(subtitles_path):
                                        s3_data_path = self.s3_manager.upload_directory(
                                            subtitles_path, 
                                            "network-analysis/input/"
                                        )
                                        logger.info(f"📤 Uploaded network data to S3: {s3_data_path}")
                                    
                                    # Use original network analysis
                                    ner = NamedEntityRecognizer()
                                    ner_df = ner.get_ners(subtitles_path, ner_path)
                                    
                                    character_network_generator = CharacterNetworkGenerator()
                                    relations_df = character_network_generator.generate_character_network(ner_df)
                                    html = character_network_generator.draw_network_graph(relations_df)
                                    
                                    # Save results to S3
                                    results_s3_path = f"network-analysis/results/network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                                    temp_path = "/tmp/network_results.html"
                                    with open(temp_path, 'w') as f:
                                        f.write(html)
                                    self.s3_manager.upload_file(temp_path, results_s3_path)
                                    logger.info(f"💾 Saved network results to S3: {results_s3_path}")
                                    
                                    return html, "✅ Network generated successfully"
                                except Exception as e:
                                    logger.error(f"❌ Network analysis error: {e}")
                                    return f"❌ Error: {str(e)}", f"❌ Failed: {str(e)}"
                            
                            get_network_button.click(
                                get_character_network_local,
                                inputs=[subtitles_path_network, ner_path],
                                outputs=[network_html, status_text]
                            )
                
                # Batch Processing Tab
                with gr.Tab("⚙️ Batch Processing"):
                    gr.Markdown("## SageMaker Batch Transform Jobs")
                    gr.Markdown("Process large datasets using SageMaker batch transform")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Create Batch Job")
                            job_name_input = gr.Textbox(label="Job Name", placeholder="batch-job-name")
                            model_name_input = gr.Textbox(label="Model Name", placeholder="deployed-model-name")
                            input_s3_input = gr.Textbox(label="Input S3 Path", placeholder="s3://bucket/input/")
                            output_s3_input = gr.Textbox(label="Output S3 Path", placeholder="s3://bucket/output/")
                            
                            create_batch_btn = gr.Button("🔄 Create Batch Job", variant="primary")
                            batch_output = gr.Markdown()
                            
                            def create_batch_job(job_name, model_name, input_path, output_path):
                                try:
                                    job_arn = self.deployment_manager.create_batch_transform_job(
                                        job_name=job_name,
                                        model_name=model_name,
                                        input_s3_path=input_path,
                                        output_s3_path=output_path
                                    )
                                    return f"✅ **Batch job created successfully!**\\n\\nJob ARN: `{job_arn}`"
                                except Exception as e:
                                    return f"❌ **Batch job failed**: {str(e)}"
                            
                            create_batch_btn.click(
                                fn=create_batch_job,
                                inputs=[job_name_input, model_name_input, input_s3_input, output_s3_input],
                                outputs=[batch_output]
                            )
                        
                        with gr.Column():
                            gr.Markdown("### Job Status")
                            status_job_input = gr.Textbox(label="Job Name to Check")
                            check_status_btn = gr.Button("📊 Check Status", variant="secondary")
                            job_status_output = gr.Markdown()
                            
                            def check_batch_status(job_name):
                                try:
                                    status = self.deployment_manager.get_batch_job_status(job_name)
                                    result = f"📊 **Job Status**: {status['status']}\\n"
                                    result += f"⏰ **Created**: {status['creation_time']}\\n"
                                    if status.get('transform_start_time'):
                                        result += f"🚀 **Started**: {status['transform_start_time']}\\n"
                                    if status.get('transform_end_time'):
                                        result += f"✅ **Completed**: {status['transform_end_time']}\\n"
                                    if status.get('failure_reason'):
                                        result += f"❌ **Failure Reason**: {status['failure_reason']}\\n"
                                    return result
                                except Exception as e:
                                    return f"❌ **Error**: {str(e)}"
                            
                            check_status_btn.click(
                                fn=check_batch_status,
                                inputs=[status_job_input],
                                outputs=[job_status_output]
                            )
        
        return app
    
    def launch(self, **kwargs):
        """Launch the Gradio application"""
        app = self.create_gradio_interface()
        
        # Default launch configuration for SageMaker
        launch_config = {
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'show_error': True,
            'share': False  # Set to True for public sharing
        }
        launch_config.update(kwargs)
        
        logger.info("🚀 Launching SageMaker-powered Gradio app...")
        logger.info(f"🌐 Server: {launch_config['server_name']}:{launch_config['server_port']}")
        logger.info("📊 Active endpoints: " + ", ".join([f"{k}={v}" for k, v in self.endpoints.items() if v]))
        
        app.launch(**launch_config)


def main():
    """Main function to launch the SageMaker Gradio app"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SageMaker-powered Stranger Things NLP Gradio App")
    parser.add_argument("--config", help="SageMaker configuration file path")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create and launch app
        app = SageMakerGradioApp(config_file=args.config)
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to launch app: {e}")
        raise


if __name__ == "__main__":
    main()
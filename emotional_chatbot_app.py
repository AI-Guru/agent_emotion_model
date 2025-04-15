#!/usr/bin/env python3
"""
Gradio web application for the Emotional Agent chatbot.

This script creates a web interface using Gradio to interact with
the EmotionalAgent, displaying both the chat and emotional state.
"""

import os
import json
import time
import logging
import argparse
import numpy as np
import gradio as gr
from typing import Dict, List, Any, Tuple
import dotenv
from openai import OpenAI

# Import our modules
from source.beammodel import BEAMProfile, EmotionState, EmotionSpectrum
from source.emotional_agent import EmotionalAgent

# Load environment variables from .env file
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("emotional_chat_app.log")
    ]
)
logger = logging.getLogger(__name__)


class EmotionalChatBot:
    """A chatbot that uses EmotionalAgent to manage its emotional state."""
    
    def __init__(
        self,
        initial_profile: BEAMProfile = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize the EmotionalChatBot.
        
        Args:
            initial_profile: Initial BEAM emotional profile
            model_name: OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens in responses
        """
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Create the emotional agent with a friendly, helpful baseline
        if initial_profile is None:
            initial_profile = BEAMProfile.create_from_emotion_names(
                sadness_joy="Serenity",  # Mild joy
                distrust_trust="Trust"    # Moderate trust
            )
        
        self.agent = EmotionalAgent(
            initial_profile=initial_profile,
            emotional_decay_rate=0.05,
            emotional_adaptation_rate=0.2,
            base_system_prompt=(
                "You are a helpful and empathetic AI assistant. "
                "Your goal is to provide accurate and useful information while connecting with the user."
            )
        )
        
        # Initialize conversation history
        self.messages = []
        
        logger.info(f"EmotionalChatBot initialized with model {model_name}")
    
    def analyze_emotions(self, text: str) -> Dict[str, float]:
        """
        Analyze emotions in text using the OpenAI API.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping BEAM spectrum names to intensity values
        """
        try:
            # Build a prompt for emotion analysis
            analysis_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are an emotional analysis assistant. Analyze the emotional content of the user's message "
                        "and provide a JSON object with emotional intensities on the following spectra:\n"
                        "- sadness_joy: -1.0 (extreme sadness) to 1.0 (extreme joy)\n"
                        "- fear_courage: -1.0 (terror) to 1.0 (heroism)\n"
                        "- distrust_trust: -1.0 (distrust) to 1.0 (admiration)\n"
                        "- anger_peace: -1.0 (rage) to 1.0 (tranquility)\n"
                        "- disgust_delight: -1.0 (loathing) to 1.0 (enthusiasm)\n"
                        "Return ONLY a valid JSON object with these fields and numerical values between -1.0 and 1.0."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
            
            # Get emotional analysis from the model
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=analysis_prompt,
                temperature=0.2,  # Lower temperature for more consistent analysis
                max_tokens=150
            )
            
            # Extract the JSON response
            analysis_text = response.choices[0].message.content.strip()
            
            # Find JSON object in the response
            start_idx = analysis_text.find('{')
            end_idx = analysis_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = analysis_text[start_idx:end_idx]
                emotions = json.loads(json_str)
            else:
                logger.warning(f"Couldn't extract JSON from analysis: {analysis_text}")
                emotions = {}
            
            # Ensure values are in the correct range
            for key, value in emotions.items():
                emotions[key] = max(-1.0, min(1.0, float(value)))
            
            logger.info(f"Emotion analysis: {emotions}")
            return emotions
            
        except Exception as e:
            logger.error(f"Error analyzing emotions: {e}")
            return {}
    
    def respond(self, user_message: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response to the user message.
        
        Args:
            user_message: The user's message
            
        Returns:
            Tuple of (assistant's response, emotional state info)
        """
        try:
            # Analyze emotions in the user message
            user_emotions = self.analyze_emotions(user_message)
            
            # Update agent's emotional state
            self.agent.update_emotional_state(emotion_adjustments=user_emotions)
            
            # Get the current system prompt
            system_prompt = self.agent.get_system_prompt()
            
            # Add user message to history
            self.messages.append({"role": "user", "content": user_message})
            
            # Prepare messages for API call
            api_messages = [{"role": "system", "content": system_prompt}] + self.messages
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response text
            assistant_message = response.choices[0].message.content
            
            # Add assistant message to history
            self.messages.append({"role": "assistant", "content": assistant_message})
            
            # Get emotion state information for display
            emotion_info = self._get_emotion_display_info()
            
            # Record the interaction
            self.agent.record_interaction(
                user_message=user_message,
                agent_response=assistant_message,
                detected_user_emotions=user_emotions
            )
            
            return assistant_message, emotion_info
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an issue while processing your message. Could you try again?", {}
    
    def _get_emotion_display_info(self) -> Dict[str, Any]:
        """
        Get formatted emotion information for display.
        
        Returns:
            Dictionary with emotion information
        """
        # Get dominant emotions
        dominant_emotions = self.agent.profile.get_dominant_emotions(n=3)
        
        # Get all emotion values as a dictionary
        emotion_values = {
            "fear_courage": self.agent.profile.fear_courage.intensity,
            "sadness_joy": self.agent.profile.sadness_joy.intensity,
            "distrust_trust": self.agent.profile.distrust_trust.intensity,
            "negligence_anticipation": self.agent.profile.negligence_anticipation.intensity,
            "anger_peace": self.agent.profile.anger_peace.intensity,
            "disgust_delight": self.agent.profile.disgust_delight.intensity,
            "disinterest_fascination": self.agent.profile.disinterest_fascination.intensity
        }
        
        # Format the dominant emotions for display
        dominant_formatted = []
        for emotion in dominant_emotions:
            formatted = f"{emotion['emotion']} ({emotion['spectrum']}: {emotion['intensity']:.2f})"
            dominant_formatted.append(formatted)
        
        return {
            "dominant_emotions": dominant_formatted,
            "emotion_values": emotion_values
        }
    
    def reset_conversation(self) -> None:
        """Reset the conversation history while maintaining emotional state."""
        self.messages = []
        logger.info("Reset conversation history")


def format_message_for_display(message):
    """Format a message for display, extracting just the text if it contains emotion data."""
    if isinstance(message, tuple) and len(message) > 0:
        return message[0]  # Extract just the text part from (text, emotion_info) tuple
    return message


def create_emotion_chart(emotion_values):
    """Create a bar chart of the emotion values using Plotly."""
    import plotly.graph_objects as go
    
    if not emotion_values:
        # Return empty chart if no values
        fig = go.Figure()
        fig.add_annotation(
            text="No emotion data available",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(title="Emotional State")
        return fig
    
    # Extract data from emotional state
    labels = list(emotion_values.keys())
    # Make the labels more readable
    labels = [label.replace("_", "-") for label in labels]
    values = list(emotion_values.values())
    
    # Create color list - negative values in red, positive in green
    colors = ["red" if v < 0 else "green" for v in values]
    
    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=colors
    ))
    
    # Update layout
    fig.update_layout(
        title="Emotional State Spectrum",
        yaxis=dict(
            title="Intensity",
            range=[-1, 1]
        ),
        xaxis=dict(
            tickangle=45
        ),
        height=400
    )
    
    return fig


def create_process_message_handler(bot):
    """Create a message processing function with a reference to the bot."""
    def process_message(user_message, history):
        """Process a user message and update the UI."""
        # Check empty input
        if not user_message.strip():
            return "", history, None, ""
        
        # Get response and emotion state
        response, emotion_info = bot.respond(user_message)
        
        # Generate chart from emotion values
        emotion_chart = create_emotion_chart(emotion_info.get("emotion_values", {}))
        
        # Format dominant emotions text
        dominant_emotions_text = "Dominant emotions: "
        if "dominant_emotions" in emotion_info and emotion_info["dominant_emotions"]:
            dominant_emotions_text += ", ".join(emotion_info["dominant_emotions"])
        else:
            dominant_emotions_text += "None"
        
        # Format the response - just get the text part if it's a tuple
        response_text = format_message_for_display(response)
        
        # Add new messages to history
        # Make a copy of the history to avoid modifying the original
        updated_history = list(history) if history else []
        updated_history.append({"role": "user", "content": user_message})
        updated_history.append({"role": "assistant", "content": response_text})
        
        # Return the necessary information
        return "", updated_history, emotion_chart, dominant_emotions_text
    
    return process_message


def create_reset_chat_handler(bot):
    """Create a reset function with a reference to the bot."""
    def reset_chat(history):
        """Reset the chat history and return a clean state."""
        import plotly.graph_objects as go
        
        bot.reset_conversation()
        
        # Create empty figure for the reset state
        fig = go.Figure()
        fig.add_annotation(
            text="No emotion data available",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(title="Emotional State")
        
        # Return empty list for the chat history
        # Empty list for the messages format
        empty_history = []
        return empty_history, fig, "Dominant emotions: None"
    
    return reset_chat


def create_ui(bot):
    """Create the Gradio UI for the chatbot."""
    with gr.Blocks(title="Emotional AI Chatbot") as app:
        gr.Markdown("# Emotional AI Chatbot")
        gr.Markdown("This chatbot uses the BEAM emotional model to simulate emotions in an AI assistant.")
        
        # Create handlers with access to the bot
        process_message_fn = create_process_message_handler(bot)
        reset_chat_fn = create_reset_chat_handler(bot)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface - explicitly setting type="messages" for gradio 5.x
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_copy_button=True,
                    type="messages"
                )
                
                # Message input
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Type your message here...",
                    lines=3,
                    max_lines=10
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Reset Chat", variant="secondary")
            
            with gr.Column(scale=1):
                # Emotion visualization
                gr.Markdown("## Agent's Emotional State")
                # Use gr.Plot which works with Plotly figures in Gradio 5.x
                emotion_chart = gr.Plot(label="Emotional State")
                dominant_emotions = gr.Markdown("Dominant emotions: None")
        
        # Event handlers
        submit_btn.click(
            fn=process_message_fn,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, emotion_chart, dominant_emotions]
        )
        
        msg.submit(
            fn=process_message_fn,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, emotion_chart, dominant_emotions]
        )
        
        clear_btn.click(
            fn=reset_chat_fn,
            inputs=[chatbot],
            outputs=[chatbot, emotion_chart, dominant_emotions]
        )
    
    return app


def main():
    """Run the Gradio application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Emotional chatbot with Gradio UI")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--share", action="store_true", help="Create a shareable link")
    
    args = parser.parse_args()
    
    # Create the chatbot
    bot = EmotionalChatBot(
        model_name=args.model,
        temperature=args.temp
    )
    
    # Create and launch the UI
    app = create_ui(bot)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
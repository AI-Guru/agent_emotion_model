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
    """A chatbot that uses EmotionalAgent to manage its emotional state with weighted history."""
    
    def __init__(
        self,
        initial_profile: BEAMProfile = None,
        model_name: str = "gpt-4.1-mini-2025-04-14",
        temperature: float = 0.7,
        max_tokens: int = 500,
        emotion_decay_factor: float = 0.8,  # How much to weight older emotions (0-1)
        max_emotion_history: int = 5        # Maximum number of previous emotions to consider
    ):
        """
        Initialize the EmotionalChatBot.
        
        Args:
            initial_profile: Initial BEAM emotional profile
            model_name: OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens in responses
            emotion_decay_factor: How much to weight older emotions (0-1)
            max_emotion_history: Maximum number of previous emotions to consider
        """
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Emotional memory parameters
        self.emotion_decay_factor = max(0.0, min(1.0, emotion_decay_factor))
        self.max_emotion_history = max(1, max_emotion_history)
        self.emotion_history = []  # List of (timestamp, emotion_dict) tuples
        
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
            
            # Add to emotional history with current timestamp
            self.add_to_emotion_history(emotions)
            
            logger.info(f"Emotion analysis: {emotions}")
            return emotions
            
        except Exception as e:
            logger.error(f"Error analyzing emotions: {e}")
            return {}
    
    def add_to_emotion_history(self, emotions: Dict[str, float]) -> None:
        """
        Add emotions to the history with current timestamp.
        
        Args:
            emotions: Dictionary of emotion intensities
        """
        # Add current emotions to history with timestamp
        current_time = time.time()
        self.emotion_history.append((current_time, emotions))
        
        # Limit the history size
        if len(self.emotion_history) > self.max_emotion_history:
            self.emotion_history = self.emotion_history[-self.max_emotion_history:]
    
    def calculate_weighted_emotions(self) -> Dict[str, float]:
        """
        Calculate weighted emotions based on history with decay.
        
        Returns:
            Dictionary of weighted emotion intensities
        """
        if not self.emotion_history:
            return {}
        
        # Initialize result with zeros
        all_emotion_keys = set()
        for _, emotions in self.emotion_history:
            all_emotion_keys.update(emotions.keys())
        
        result = {key: 0.0 for key in all_emotion_keys}
        
        # Calculate weights for each history entry
        total_weight = 0.0
        current_weight = 1.0  # Most recent has weight 1.0
        
        # Process from most recent to oldest
        for _, emotions in reversed(self.emotion_history):
            # Apply weights to each emotion
            for key, value in emotions.items():
                result[key] += value * current_weight
            
            # Track total weight applied
            total_weight += current_weight
            
            # Decay weight for older entries
            current_weight *= self.emotion_decay_factor
        
        # Normalize by total weight
        if total_weight > 0:
            for key in result:
                result[key] /= total_weight
        
        return result
    
    def respond(self, user_message: str, chat_history: List = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response to the user message.
        
        Args:
            user_message: The user's message
            chat_history: Optional chat history from the UI
            
        Returns:
            Tuple of (assistant's response, emotional state info)
        """
        try:
            # Analyze emotions in the user message
            user_emotions = self.analyze_emotions(user_message)
            
            # Calculate weighted emotions from history
            weighted_emotions = self.calculate_weighted_emotions()
            
            # Log the weighted emotions
            logger.info(f"Weighted emotions across conversation: {weighted_emotions}")
            
            # Update agent's emotional state with weighted emotions
            # This gives a more stable emotional response that considers conversation history
            self.agent.update_emotional_state(emotion_adjustments=weighted_emotions)
            
            # Get the current system prompt
            system_prompt = self.agent.get_system_prompt()
            
            # Prepare messages for API call, starting with the system prompt
            api_messages = [{"role": "system", "content": system_prompt}]
            
            # Add chat history if provided (from the UI)
            if chat_history:
                # Convert the UI history format to OpenAI API format
                for message in chat_history:
                    if isinstance(message, dict) and "role" in message and "content" in message:
                        api_messages.append(message)
            
            # Add the current user message
            api_messages.append({"role": "user", "content": user_message})
            
            # Update our internal history
            self.messages = api_messages.copy()
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response text
            assistant_message = response.choices[0].message.content
            
            # Add assistant message to our internal history
            self.messages.append({"role": "assistant", "content": assistant_message})
            
            # Get emotion state information for display
            emotion_info = self._get_emotion_display_info()
            
            # Record the interaction
            self.agent.record_interaction(
                user_message=user_message,
                agent_response=assistant_message,
                detected_user_emotions=weighted_emotions  # Use weighted emotions
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
    """Create a BEAM emotion chart similar to beam.jpg using Plotly."""
    import plotly.graph_objects as go
    from source.beammodel import EMOTION_MAPPINGS, EmotionSpectrum
    
    if not emotion_values:
        # Return empty chart if no values
        fig = go.Figure()
        fig.add_annotation(
            text="No emotion data available",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(title="BEAM Emotional Profile")
        return fig
    
    # Define the intensity values that match beam.jpg
    intensities = [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]
    intensity_labels = [str(i) for i in intensities]
    
    # Define BEAM spectrum colors - made more vibrant and closer to beam.jpg
    spectrum_colors = {
        "fear_courage": "rgba(255, 80, 80, 0.8)",         # Brighter Red
        "sadness_joy": "rgba(255, 180, 60, 0.8)",        # Brighter Orange
        "distrust_trust": "rgba(255, 255, 60, 0.8)",     # Brighter Yellow
        "negligence_anticipation": "rgba(100, 255, 100, 0.8)",  # Brighter Green
        "anger_peace": "rgba(80, 200, 255, 0.8)",        # Sky Blue
        "disgust_delight": "rgba(100, 100, 255, 0.8)",   # Brighter Blue
        "disinterest_fascination": "rgba(180, 100, 255, 0.8)"  # Brighter Purple
    }
    
    # Create a figure
    fig = go.Figure()
    
    # Define the order of spectra to match beam.jpg
    spectra_order = [
        "fear_courage", 
        "sadness_joy", 
        "distrust_trust", 
        "negligence_anticipation", 
        "anger_peace", 
        "disgust_delight", 
        "disinterest_fascination"
    ]
    
    # Add horizontal bars for each spectrum with markers at current value
    y_positions = []
    current_y = 7
    
    for spectrum_name in spectra_order:
        current_y -= 1
        y_positions.append(current_y)
        
        # Get the current value for this spectrum
        current_value = emotion_values.get(spectrum_name, 0)
        
        # Add a line representing the spectrum
        fig.add_trace(go.Scatter(
            x=intensities,
            y=[current_y] * len(intensities),
            mode='lines',
            line=dict(color=spectrum_colors[spectrum_name], width=12),
            name=spectrum_name.replace('_', '-'),
            showlegend=False
        ))
        
        # Add marker for current value
        fig.add_trace(go.Scatter(
            x=[current_value],
            y=[current_y],
            mode='markers',
            marker=dict(color='black', size=14, line=dict(color='white', width=2)),
            name=f"Current: {spectrum_name.replace('_', '-')}",
            showlegend=False
        ))
        
        # Get the current emotion name for this spectrum
        current_intensity_levels = list(EMOTION_MAPPINGS[getattr(EmotionSpectrum, spectrum_name.upper())].keys())
        closest_intensity = min(current_intensity_levels, key=lambda x: abs(x - current_value))
        current_emotion = EMOTION_MAPPINGS[getattr(EmotionSpectrum, spectrum_name.upper())][closest_intensity]
        
        # Add spectrum title with current emotion
        formatted_name = spectrum_name.replace("_", "-").title()
        
        # Add emotion words for each position
        for i, intensity in enumerate(intensities):
            emotion_name = EMOTION_MAPPINGS[getattr(EmotionSpectrum, spectrum_name.upper())][intensity]
            
            # Determine text style - bolder and black for the closest emotion to current value
            text_color = 'black'
            text_size = 12
            
            fig.add_annotation(
                x=intensity,
                y=current_y + 0.2,  # Position above the line
                text=emotion_name,
                showarrow=False,
                font=dict(color=text_color, size=text_size, family="Arial"),
                xanchor='center',
                yanchor='bottom'
            )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "BEAM Emotional Profile",
            'font': {'size': 22}
        },
        xaxis=dict(
            title="",  # Remove the "Intensity" title
            tickvals=intensities,
            ticktext=[""] * len(intensities),  # Empty strings for tick labels
            range=[-1.1, 1.1],
            tickfont={'size': 1, 'color': 'rgba(0,0,0,0)'}  # Invisible tick labels
        ),
        yaxis=dict(
            showticklabels=False,
            range=[-0.5, 7.5]
        ),
        height=700,  # Increased height
        margin=dict(l=20, r=20, t=60, b=50),
        plot_bgcolor='white',
        showlegend=False,  # Hide the legend completely
        annotations=[]  # Start with empty annotations list to avoid duplicates
    )
    
    # Add gray gridlines at each intensity value
    for intensity in intensities:
        fig.add_shape(
            type="line",
            x0=intensity,
            x1=intensity,
            y0=-0.5,
            y1=7.5,
            line=dict(color="rgba(200,200,200,0.5)", width=1, dash="dot")
        )
    
    return fig


def create_process_message_handler(bot):
    """Create a message processing function with a reference to the bot."""
    def process_message(user_message, history):
        """Process a user message and update the UI."""
        # Check empty input
        if not user_message.strip():
            return "", history, None, ""
        
        # Convert history to a list if it's not already (for safety)
        history_list = list(history) if history else []
        
        # Get response and emotion state - pass the chat history
        response, emotion_info = bot.respond(user_message, history_list)
        
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
        fig.update_layout(title="BEAM Emotional Profile")
        
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
                gr.Markdown("## Conversation Emotional Profile")
                # Use gr.Plot which works with Plotly figures in Gradio 5.x
                emotion_chart = gr.Plot(label="BEAM Emotional Profile")
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
    parser.add_argument("--model", type=str, default="gpt-4.1-mini-2025-04-14", help="OpenAI model to use")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--share", action="store_true", help="Create a shareable link")
    parser.add_argument("--decay", type=float, default=0.8, help="Emotion decay factor (0-1)")
    parser.add_argument("--history", type=int, default=5, help="Number of messages to consider for emotion")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="The IP address to bind the server to")
    
    args = parser.parse_args()
    
    # Create the chatbot
    bot = EmotionalChatBot(
        model_name=args.model,
        temperature=args.temp,
        emotion_decay_factor=args.decay,
        max_emotion_history=args.history
    )
    
    # Create and launch the UI
    app = create_ui(bot)
    app.launch(server_name=args.server_name, share=args.share)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Example of integrating the EmotionalAgent with an LLM API.

This script demonstrates how to use the EmotionalAgent with OpenAI's API
to create an emotionally aware conversational agent.

Note: This is a demonstration script and requires:
1. OpenAI API key as an environment variable OPENAI_API_KEY
2. The beammodel.py and emotional_prompt_generator.py modules
3. The emotional_agent.py module
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
import dotenv

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
        logging.FileHandler("emotional_chat.log")
    ]
)
logger = logging.getLogger(__name__)


class EmotionalChatBot:
    """A chatbot that uses EmotionalAgent to manage its emotional state."""
    
    def __init__(
        self,
        initial_profile: Optional[BEAMProfile] = None,
        model_name: str = "gpt-4.1-mini-2025-04-14",
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
        
        # Create the emotional agent
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
    
    def respond(self, user_message: str) -> str:
        """
        Generate a response to the user message.
        
        Args:
            user_message: The user's message
            
        Returns:
            The assistant's response
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
            
            # Record the interaction
            self.agent.record_interaction(
                user_message=user_message,
                agent_response=assistant_message,
                detected_user_emotions=user_emotions
            )
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an issue while processing your message. Could you try again?"
    
    def save_conversation(self, filepath: str) -> None:
        """
        Save the current conversation to a file.
        
        Args:
            filepath: Path to save the conversation
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.messages, f, indent=2)
            logger.info(f"Saved conversation to {filepath}")
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
    def reset_conversation(self) -> None:
        """Reset the conversation history while maintaining emotional state."""
        self.messages = []
        logger.info("Reset conversation history")


def interactive_chat():
    """Run an interactive chat session with the EmotionalChatBot."""
    
    # Create a parser for command-line arguments
    parser = argparse.ArgumentParser(description="Interactive chat with an emotionally aware AI assistant")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini-2025-04-14", help="OpenAI model to use")
    parser.add_argument("--profile", type=str, help="Path to a JSON file with initial emotional profile")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for response generation")
    
    args = parser.parse_args()
    
    # Load initial profile if specified
    initial_profile = None
    if args.profile:
        try:
            with open(args.profile, 'r') as f:
                profile_data = json.load(f)
            
            initial_profile = BEAMProfile()
            for spectrum, intensity in profile_data.items():
                initial_profile.adjust_emotion(spectrum, float(intensity))
                
            logger.info(f"Loaded initial profile from {args.profile}")
        except Exception as e:
            logger.error(f"Error loading profile: {e}")
            initial_profile = None
    
    # Create the chatbot
    chatbot = EmotionalChatBot(
        initial_profile=initial_profile,
        model_name=args.model,
        temperature=args.temperature
    )
    
    print("\n=== Emotional ChatBot ===")
    print("Type 'exit' to end the conversation, 'save' to save the conversation,")
    print("or 'reset' to start a new conversation.")
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
                
            elif user_input.lower() == 'save':
                filename = f"conversation_{int(time.time())}.json"
                chatbot.save_conversation(filename)
                print(f"Conversation saved to {filename}")
                continue
                
            elif user_input.lower() == 'reset':
                chatbot.reset_conversation()
                print("Started a new conversation.")
                continue
            
            if not user_input:
                continue
                
            print("\nAssistant: ", end="", flush=True)
            response = chatbot.respond(user_input)
            print(response)
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"Error in chat loop: {e}")
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    interactive_chat()
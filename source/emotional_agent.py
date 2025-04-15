#!/usr/bin/env python3
"""
EmotionalAgent class for integrating BEAM emotional states with LLM interactions.

This module provides a class that manages the emotional state of an LLM-based agent
and generates appropriate system prompts based on the emotional profile.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable

# Import BEAM model
from source.beammodel import BEAMProfile, EmotionState, EmotionSpectrum
from source.emotional_prompt_generator import generate_emotional_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmotionalAgent:
    """A class for managing an LLM agent with emotional states using the BEAM model."""
    
    def __init__(
        self, 
        initial_profile: Optional[BEAMProfile] = None,
        emotional_decay_rate: float = 0.05,
        emotional_adaptation_rate: float = 0.1,
        base_system_prompt: str = "You are a helpful assistant."
    ):
        """
        Initialize the EmotionalAgent.
        
        Args:
            initial_profile: Initial BEAM emotional profile (defaults to neutral)
            emotional_decay_rate: Rate at which emotions decay back to baseline (0.0 to 1.0)
            emotional_adaptation_rate: Rate at which the agent adapts to detected user emotions
            base_system_prompt: Base system prompt to append emotional guidelines to
        """
        self.profile = initial_profile if initial_profile else BEAMProfile()
        self.baseline_profile = self.profile.model_copy(deep=True)
        self.emotional_decay_rate = max(0.0, min(1.0, emotional_decay_rate))
        self.emotional_adaptation_rate = max(0.0, min(1.0, emotional_adaptation_rate))
        self.base_system_prompt = base_system_prompt
        self.last_update_time = time.time()
        self.interaction_history = []
        
        # Generate initial prompt
        self.current_prompt = self._generate_prompt()
        
        logger.info("Emotional agent initialized")
    
    def _generate_prompt(self) -> str:
        """Generate a system prompt based on the current emotional profile."""
        emotional_guidelines = generate_emotional_prompt(self.profile)
        return f"{self.base_system_prompt}\n\n{emotional_guidelines}"
    
    def update_emotional_state(self, 
                              emotion_adjustments: Optional[Dict[str, float]] = None, 
                              user_message: Optional[str] = None,
                              emotion_analyzer: Optional[Callable[[str], Dict[str, float]]] = None) -> None:
        """
        Update the agent's emotional state based on direct adjustments or user message analysis.
        
        Args:
            emotion_adjustments: Direct adjustments to emotion spectra (e.g., {"sadness_joy": 0.2})
            user_message: User message to analyze for emotional content
            emotion_analyzer: Function to analyze emotions in user messages
        """
        # Apply time-based emotional decay toward baseline
        current_time = time.time()
        time_since_update = current_time - self.last_update_time
        decay_amount = self.emotional_decay_rate * time_since_update
        
        if decay_amount > 0:
            # Create a blend that moves toward baseline
            decay_weight = min(decay_amount, 1.0)
            self.profile = self.profile.blend_with(self.baseline_profile, weight=decay_weight)
            logger.debug(f"Applied emotional decay with weight {decay_weight}")
        
        # Apply direct emotion adjustments
        if emotion_adjustments:
            for spectrum_name, adjustment in emotion_adjustments.items():
                self.profile.adjust_emotion(spectrum_name, adjustment)
                logger.debug(f"Adjusted {spectrum_name} by {adjustment}")
        
        # Analyze user message for emotional content if provided
        if user_message and emotion_analyzer:
            try:
                detected_emotions = emotion_analyzer(user_message)
                
                # Create a temporary profile from detected emotions
                detected_profile = BEAMProfile()
                for spectrum_name, intensity in detected_emotions.items():
                    detected_profile.adjust_emotion(spectrum_name, intensity)
                
                # Adapt the agent's profile toward detected emotions
                self.profile = self.profile.blend_with(
                    detected_profile, 
                    weight=self.emotional_adaptation_rate
                )
                logger.debug(f"Adapted to user emotions with weight {self.emotional_adaptation_rate}")
                
            except Exception as e:
                logger.error(f"Error analyzing user message: {e}")
        
        # Update the prompt based on the new emotional state
        self.current_prompt = self._generate_prompt()
        self.last_update_time = current_time
        logger.info("Updated emotional state and prompt")
    
    def get_system_prompt(self) -> str:
        """Get the current system prompt with emotional guidelines."""
        return self.current_prompt
    
    def record_interaction(self, 
                          user_message: str, 
                          agent_response: str, 
                          detected_user_emotions: Optional[Dict[str, float]] = None,
                          agent_emotions: Optional[Dict[str, float]] = None) -> None:
        """
        Record an interaction for history and analysis.
        
        Args:
            user_message: The message from the user
            agent_response: The agent's response
            detected_user_emotions: Detected emotions in the user message
            agent_emotions: The agent's emotions during response generation
        """
        interaction = {
            "timestamp": time.time(),
            "user_message": user_message,
            "agent_response": agent_response,
            "detected_user_emotions": detected_user_emotions,
            "agent_emotions": agent_emotions or self.profile.to_vector().tolist()
        }
        self.interaction_history.append(interaction)
        logger.debug("Recorded interaction")
    
    def set_baseline_profile(self, profile: BEAMProfile) -> None:
        """
        Set a new baseline emotional profile.
        
        Args:
            profile: The new baseline BEAM profile
        """
        self.baseline_profile = profile.model_copy(deep=True)
        logger.info("Updated baseline emotional profile")
    
    def save_state(self, filepath: str) -> None:
        """
        Save the agent's current state to a file.
        
        Args:
            filepath: Path to save the state file
        """
        state = {
            "current_profile": self.profile.model_dump(),
            "baseline_profile": self.baseline_profile.model_dump(),
            "emotional_decay_rate": self.emotional_decay_rate,
            "emotional_adaptation_rate": self.emotional_adaptation_rate,
            "base_system_prompt": self.base_system_prompt,
            "last_update_time": self.last_update_time,
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved agent state to {filepath}")
        except Exception as e:
            logger.error(f"Error saving agent state: {e}")
    
    @classmethod
    def load_state(cls, filepath: str) -> "EmotionalAgent":
        """
        Load an agent state from a file.
        
        Args:
            filepath: Path to the state file
            
        Returns:
            An EmotionalAgent with the loaded state
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Create profiles from the state
            current_profile = BEAMProfile.model_validate(state["current_profile"])
            baseline_profile = BEAMProfile.model_validate(state["baseline_profile"])
            
            # Create the agent
            agent = cls(
                initial_profile=current_profile,
                emotional_decay_rate=state["emotional_decay_rate"],
                emotional_adaptation_rate=state["emotional_adaptation_rate"],
                base_system_prompt=state["base_system_prompt"]
            )
            
            # Set additional state properties
            agent.baseline_profile = baseline_profile
            agent.last_update_time = state["last_update_time"]
            
            logger.info(f"Loaded agent state from {filepath}")
            return agent
            
        except Exception as e:
            logger.error(f"Error loading agent state: {e}")
            return cls()  # Return default agent if loading fails


# Example usage
if __name__ == "__main__":
    # Create a sample emotion analyzer function
    def simple_emotion_analyzer(text: str) -> Dict[str, float]:
        """A very simple emotion analyzer that looks for key words."""
        emotions = {
            "sadness_joy": 0.0,
            "fear_courage": 0.0,
            "distrust_trust": 0.0
        }
        
        # Simple keyword matching (just for demonstration)
        if any(word in text.lower() for word in ["happy", "glad", "excited", "joy"]):
            emotions["sadness_joy"] = 0.5
        if any(word in text.lower() for word in ["sad", "unhappy", "disappointed"]):
            emotions["sadness_joy"] = -0.5
        if any(word in text.lower() for word in ["afraid", "scared", "worried"]):
            emotions["fear_courage"] = -0.5
        if any(word in text.lower() for word in ["confident", "brave", "sure"]):
            emotions["fear_courage"] = 0.5
        if any(word in text.lower() for word in ["trust", "believe", "reliable"]):
            emotions["distrust_trust"] = 0.5
        if any(word in text.lower() for word in ["doubt", "suspect", "unsure"]):
            emotions["distrust_trust"] = -0.5
            
        return emotions
    
    # Create an initial profile - a friendly, joyful assistant
    initial_profile = BEAMProfile.create_from_emotion_names(
        sadness_joy="Joy",
        distrust_trust="Trust"
    )
    
    # Create the agent
    agent = EmotionalAgent(
        initial_profile=initial_profile,
        emotional_decay_rate=0.1,
        emotional_adaptation_rate=0.3,
        base_system_prompt="You are a helpful AI assistant designed to provide information and assistance."
    )
    
    # Print the initial prompt
    print("=== Initial System Prompt ===")
    print(agent.get_system_prompt())
    
    # Simulate an interaction
    user_message = "I'm feeling a bit worried about my presentation tomorrow."
    print(f"\n=== User Message ===\n{user_message}")
    
    # Update emotional state based on user message
    agent.update_emotional_state(
        user_message=user_message,
        emotion_analyzer=simple_emotion_analyzer
    )
    
    # Print the updated prompt
    print("\n=== Updated System Prompt ===")
    print(agent.get_system_prompt())
    
    # Save the agent state
    agent.save_state("agent_state.json")
    print("\nSaved agent state to agent_state.json")
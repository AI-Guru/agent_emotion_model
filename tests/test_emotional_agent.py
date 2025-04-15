#!/usr/bin/env python3
"""
Tests for the EmotionalAgent module.
"""

import pytest
import json
import time
import os
from typing import Dict
from source.beammodel import BEAMProfile
from source.emotional_agent import EmotionalAgent


def test_agent_initialization():
    """Test creating an EmotionalAgent with default and custom profiles."""
    # Test with default profile
    default_agent = EmotionalAgent()
    assert default_agent.profile is not None
    assert default_agent.profile.sadness_joy.intensity == 0.0
    
    # Test with custom profile
    initial_profile = BEAMProfile.create_from_emotion_names(
        sadness_joy="Joy",
        distrust_trust="Trust"
    )
    custom_agent = EmotionalAgent(
        initial_profile=initial_profile,
        emotional_decay_rate=0.1,
        emotional_adaptation_rate=0.3,
        base_system_prompt="Custom prompt."
    )
    
    assert custom_agent.profile.sadness_joy.intensity == 0.6
    assert custom_agent.profile.distrust_trust.intensity == 0.6
    assert custom_agent.emotional_decay_rate == 0.1
    assert custom_agent.emotional_adaptation_rate == 0.3
    assert custom_agent.base_system_prompt == "Custom prompt."


def test_generate_system_prompt():
    """Test generating a system prompt from the emotional profile."""
    initial_profile = BEAMProfile.create_from_emotion_names(
        sadness_joy="Joy",
        distrust_trust="Trust"
    )
    agent = EmotionalAgent(
        initial_profile=initial_profile,
        base_system_prompt="You are a helpful AI assistant."
    )
    
    # Get the prompt
    prompt = agent.get_system_prompt()
    
    # Check that it contains the base system prompt
    assert "You are a helpful AI assistant." in prompt
    
    # Check that it contains emotion info
    assert "Joy" in prompt
    assert "Trust" in prompt


def test_emotional_state_update_direct():
    """Test updating emotional state with direct adjustments."""
    agent = EmotionalAgent()
    
    # Initial state should be neutral
    assert agent.profile.sadness_joy.intensity == 0.0
    
    # Update emotions directly
    agent.update_emotional_state(emotion_adjustments={"sadness_joy": 0.5})
    
    # Check that emotion was updated
    assert agent.profile.sadness_joy.intensity == 0.5
    
    # Another update should add to the existing value
    agent.update_emotional_state(emotion_adjustments={"sadness_joy": 0.2})
    # The value should be approximately 0.7 (may have small floating-point differences)
    assert pytest.approx(agent.profile.sadness_joy.intensity, 0.001) == 0.7


def test_emotional_analyzer_integration():
    """Test integration with an emotion analyzer function."""
    agent = EmotionalAgent()
    
    # Create a sample emotion analyzer function
    def simple_emotion_analyzer(text: str) -> Dict[str, float]:
        """A very simple emotion analyzer that looks for key words."""
        emotions = {
            "sadness_joy": 0.0,
            "fear_courage": 0.0,
            "distrust_trust": 0.0
        }
        
        if "happy" in text.lower():
            emotions["sadness_joy"] = 0.5
        if "worried" in text.lower():
            emotions["fear_courage"] = -0.5
            
        return emotions
    
    # Update emotional state based on user message
    user_message = "I'm feeling a bit worried about my presentation, but happy about the weekend."
    agent.update_emotional_state(
        user_message=user_message,
        emotion_analyzer=simple_emotion_analyzer
    )
    
    # Check that emotions were updated
    assert agent.profile.sadness_joy.intensity > 0  # Moved toward joy
    assert agent.profile.fear_courage.intensity < 0  # Moved toward fear


def test_emotional_decay():
    """Test the emotional decay mechanism."""
    # Create a test that directly tests the blend functionality used by the decay mechanism
    initial_profile = BEAMProfile()
    initial_profile.adjust_emotion("sadness_joy", 1.0)  # Set to extreme joy
    
    baseline_profile = BEAMProfile()  # Default neutral profile
    
    # Manually blend the profiles as the decay function would
    # A decay weight of 0.5 means 50% decay toward baseline
    decay_weight = 0.5
    blended_profile = initial_profile.blend_with(baseline_profile, weight=decay_weight)
    
    # The result should be intermediate between the initial and baseline values
    # Initial: 1.0, Baseline: 0.0, Expected after 50% decay: 0.5
    assert blended_profile.sadness_joy.intensity < 1.0
    assert pytest.approx(blended_profile.sadness_joy.intensity, 0.01) == 0.5


def test_record_interaction():
    """Test recording an interaction."""
    agent = EmotionalAgent()
    
    # Record an interaction
    agent.record_interaction(
        user_message="Hello",
        agent_response="Hi there",
        detected_user_emotions={"sadness_joy": 0.3},
        agent_emotions={"sadness_joy": 0.2}
    )
    
    # Check that interaction was recorded
    assert len(agent.interaction_history) == 1
    interaction = agent.interaction_history[0]
    assert interaction["user_message"] == "Hello"
    assert interaction["agent_response"] == "Hi there"
    assert interaction["detected_user_emotions"] == {"sadness_joy": 0.3}
    assert interaction["agent_emotions"] == {"sadness_joy": 0.2}


def test_save_and_load_state(tmp_path):
    """Test saving and loading agent state."""
    # Create an agent with custom settings
    initial_profile = BEAMProfile.create_from_emotion_names(
        sadness_joy="Joy",
        distrust_trust="Trust"
    )
    agent = EmotionalAgent(
        initial_profile=initial_profile,
        emotional_decay_rate=0.1,
        emotional_adaptation_rate=0.3,
        base_system_prompt="Custom prompt."
    )
    
    # Save state to a temporary file
    filepath = tmp_path / "test_state.json"
    agent.save_state(filepath)
    
    # Verify file exists
    assert os.path.exists(filepath)
    
    # Load state into a new agent
    loaded_agent = EmotionalAgent.load_state(filepath)
    
    # Verify loaded state matches original
    assert loaded_agent.profile.sadness_joy.intensity == agent.profile.sadness_joy.intensity
    assert loaded_agent.profile.distrust_trust.intensity == agent.profile.distrust_trust.intensity
    assert loaded_agent.emotional_decay_rate == agent.emotional_decay_rate
    assert loaded_agent.emotional_adaptation_rate == agent.emotional_adaptation_rate
    assert loaded_agent.base_system_prompt == agent.base_system_prompt
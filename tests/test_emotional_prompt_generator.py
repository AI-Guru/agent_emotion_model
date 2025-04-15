#!/usr/bin/env python3
"""
Tests for the emotional prompt generator module.
"""

import pytest
import json
import os
from source.beammodel import BEAMProfile, EmotionState, EmotionSpectrum
from source.emotional_prompt_generator import (
    generate_emotional_prompt,
    extract_primary_secondary_emotions,
    format_emotion_description,
    get_linguistic_features,
    generate_adaptation_rules,
    create_beam_profile_from_json
)


def test_extract_primary_secondary_emotions():
    """Test extracting primary and secondary emotions from a profile."""
    # Create a profile with varying intensities
    profile = BEAMProfile(
        sadness_joy=EmotionState(spectrum=EmotionSpectrum.SADNESS_JOY, intensity=0.8),
        fear_courage=EmotionState(spectrum=EmotionSpectrum.FEAR_COURAGE, intensity=0.3),
        distrust_trust=EmotionState(spectrum=EmotionSpectrum.DISTRUST_TRUST, intensity=-0.6)
    )
    
    result = extract_primary_secondary_emotions(profile)
    
    # Check structure
    assert "primary" in result
    assert "secondary" in result
    
    # Primary should be sadness_joy (highest absolute intensity)
    assert result["primary"]["spectrum"] == "sadness_joy"
    # The emotion at 0.8 intensity could be Joy or Ecstasy depending on the exact mapping
    assert result["primary"]["emotion"] in ["Joy", "Ecstasy"]
    assert result["primary"]["intensity"] == 0.8
    
    # Secondary should be distrust_trust (second highest absolute intensity)
    assert result["secondary"]["spectrum"] == "distrust_trust"
    assert result["secondary"]["emotion"] == "Wary"
    assert result["secondary"]["intensity"] == -0.6


def test_format_emotion_description():
    """Test formatting emotion data into natural language descriptions."""
    emotion_dict = {
        "spectrum": "fear_courage",
        "emotion": "Courage",
        "intensity": 0.6
    }
    
    result = format_emotion_description(emotion_dict)
    
    # Check result
    assert result["emotion"] == "Courage"
    assert result["intensity"] == "strong"
    assert result["direction"] == "courage"
    assert result["raw_intensity"] == 0.6
    
    # Test with negative intensity
    emotion_dict = {
        "spectrum": "sadness_joy",
        "emotion": "Sadness",
        "intensity": -0.6
    }
    
    result = format_emotion_description(emotion_dict)
    assert result["intensity"] == "strong"
    assert result["direction"] == "sadness"


def test_get_linguistic_features():
    """Test generating linguistic feature suggestions based on emotion."""
    emotion_dict = {
        "emotion": "Joy",
        "intensity": "strong",
        "direction": "joy",
        "raw_intensity": 0.6
    }
    
    features = get_linguistic_features(emotion_dict)
    
    # Check result contains expected guidance for joy
    assert "use positive language" in features.lower()
    assert "express enthusiasm" in features.lower()
    
    # Test with different emotion
    emotion_dict = {
        "emotion": "Fear",
        "intensity": "moderate",
        "direction": "fear",
        "raw_intensity": -0.6
    }
    
    features = get_linguistic_features(emotion_dict)
    
    # Check result contains expected guidance for fear
    assert "express concern" in features.lower()
    assert "cautious language" in features.lower()


def test_generate_adaptation_rules():
    """Test generating adaptation rules based on emotions."""
    primary = {
        "emotion": "Joy", 
        "intensity": "strong",
        "direction": "joy",
        "raw_intensity": 0.6
    }
    
    secondary = {
        "emotion": "Trust",
        "intensity": "moderate",
        "direction": "trust",
        "raw_intensity": 0.5
    }
    
    rules = generate_adaptation_rules(primary, secondary)
    
    # Check result is a string
    assert isinstance(rules, str)
    
    # Check it contains the common adaptation points
    assert "When helping with problem-solving" in rules
    
    # Check it contains specific adaptations for the emotions
    assert "When discussing serious topics" in rules  # Joy adaptation
    assert "When addressing uncertainties" in rules  # Trust adaptation


def test_create_beam_profile_from_json():
    """Test creating a BEAM profile from JSON data."""
    json_data = {
        "sadness_joy": 0.6,
        "distrust_trust": -0.3,
        "fear_courage": 0.0
    }
    
    profile = create_beam_profile_from_json(json_data)
    
    # Check profile was created correctly
    assert profile.sadness_joy.intensity == 0.6
    assert profile.distrust_trust.intensity == -0.3
    assert profile.fear_courage.intensity == 0.0


def test_generate_emotional_prompt():
    """Test generating a complete emotional prompt."""
    # Create a profile
    profile = BEAMProfile(
        sadness_joy=EmotionState(spectrum=EmotionSpectrum.SADNESS_JOY, intensity=0.6),
        distrust_trust=EmotionState(spectrum=EmotionSpectrum.DISTRUST_TRUST, intensity=0.6)
    )
    
    prompt = generate_emotional_prompt(profile)
    
    # Check prompt is a string
    assert isinstance(prompt, str)
    
    # Check it contains the key sections
    assert "You are an assistant with the following emotional characteristics" in prompt
    assert "Primary:" in prompt
    assert "Secondary:" in prompt
    assert "Express these emotions naturally" in prompt
    assert "Adapt your emotional expression" in prompt
    
    # Check it mentions the emotions
    assert "Joy" in prompt
    assert "Trust" in prompt
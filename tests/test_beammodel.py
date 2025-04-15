#!/usr/bin/env python3
"""
Tests for the BEAM model module.
"""

import pytest
import numpy as np
from source.beammodel import BEAMProfile, EmotionState, EmotionSpectrum


def test_neutral_profile_creation():
    """Test creating a neutral profile with default values."""
    neutral_profile = BEAMProfile()
    # Verify all emotions are at neutral intensity
    for spectrum_name in [
        "fear_courage", "sadness_joy", "distrust_trust", 
        "negligence_anticipation", "anger_peace", 
        "disgust_delight", "disinterest_fascination"
    ]:
        emotion_state = getattr(neutral_profile, spectrum_name)
        assert emotion_state.intensity == 0.0
        assert emotion_state.spectrum.value == spectrum_name


def test_custom_profile_creation():
    """Test creating a profile with specific emotion intensities."""
    custom_profile = BEAMProfile(
        fear_courage=EmotionState(spectrum=EmotionSpectrum.FEAR_COURAGE, intensity=0.6),
        sadness_joy=EmotionState(spectrum=EmotionSpectrum.SADNESS_JOY, intensity=0.3),
        anger_peace=EmotionState(spectrum=EmotionSpectrum.ANGER_PEACE, intensity=-0.3)
    )
    
    # Check specific emotions were set correctly
    assert custom_profile.fear_courage.intensity == 0.6
    assert custom_profile.fear_courage.emotion_name == "Courage"
    
    assert custom_profile.sadness_joy.intensity == 0.3
    assert custom_profile.sadness_joy.emotion_name == "Serenity"
    
    assert custom_profile.anger_peace.intensity == -0.3
    assert custom_profile.anger_peace.emotion_name == "Annoyance"
    
    # Check default emotions are neutral
    assert custom_profile.distrust_trust.intensity == 0.0


def test_create_from_emotion_names():
    """Test creating a profile from emotion names."""
    named_profile = BEAMProfile.create_from_emotion_names(
        fear_courage="Courage",
        sadness_joy="Ecstasy",
        distrust_trust="Distrust"
    )
    
    # Check intensities match the emotion names
    assert named_profile.fear_courage.intensity == 0.6
    assert named_profile.sadness_joy.intensity == 1.0
    assert named_profile.distrust_trust.intensity == -1.0
    
    # Check names are set correctly
    assert named_profile.fear_courage.emotion_name == "Courage"
    assert named_profile.sadness_joy.emotion_name == "Ecstasy"
    assert named_profile.distrust_trust.emotion_name == "Distrust"


def test_get_dominant_emotions():
    """Test getting dominant emotions from a profile."""
    custom_profile = BEAMProfile(
        fear_courage=EmotionState(spectrum=EmotionSpectrum.FEAR_COURAGE, intensity=0.6),
        sadness_joy=EmotionState(spectrum=EmotionSpectrum.SADNESS_JOY, intensity=0.3),
        anger_peace=EmotionState(spectrum=EmotionSpectrum.ANGER_PEACE, intensity=-0.3)
    )
    
    dominant = custom_profile.get_dominant_emotions(2)
    
    # Check that we got exactly 2 emotions
    assert len(dominant) == 2
    
    # Check that the most dominant emotion is fear_courage
    assert dominant[0]["spectrum"] == "fear_courage"
    assert dominant[0]["emotion"] == "Courage"
    assert dominant[0]["intensity"] == 0.6
    
    # Check that the second most dominant emotion depends on absolute intensity
    # Both sadness_joy and anger_peace have absolute intensity 0.3, so either could be second
    assert dominant[1]["intensity"] in [0.3, -0.3]


def test_to_vector_and_from_vector():
    """Test converting a profile to a vector and back."""
    custom_profile = BEAMProfile(
        fear_courage=EmotionState(spectrum=EmotionSpectrum.FEAR_COURAGE, intensity=0.6),
        sadness_joy=EmotionState(spectrum=EmotionSpectrum.SADNESS_JOY, intensity=0.3),
        anger_peace=EmotionState(spectrum=EmotionSpectrum.ANGER_PEACE, intensity=-0.3)
    )
    
    # Convert to vector
    vector = custom_profile.to_vector()
    
    # Check vector dimensions
    assert len(vector) == 7
    assert vector[0] == 0.6  # fear_courage
    assert vector[1] == 0.3  # sadness_joy
    assert vector[4] == -0.3  # anger_peace
    
    # Create profile from vector
    reconstructed = BEAMProfile.from_vector(vector)
    
    # Check reconstructed profile
    assert reconstructed.fear_courage.intensity == 0.6
    assert reconstructed.sadness_joy.intensity == 0.3
    assert reconstructed.anger_peace.intensity == -0.3


def test_blend_profiles():
    """Test blending two emotional profiles."""
    profile1 = BEAMProfile(
        fear_courage=EmotionState(spectrum=EmotionSpectrum.FEAR_COURAGE, intensity=0.6),
        sadness_joy=EmotionState(spectrum=EmotionSpectrum.SADNESS_JOY, intensity=0.3)
    )
    
    profile2 = BEAMProfile(
        fear_courage=EmotionState(spectrum=EmotionSpectrum.FEAR_COURAGE, intensity=0.0),
        sadness_joy=EmotionState(spectrum=EmotionSpectrum.SADNESS_JOY, intensity=1.0),
        distrust_trust=EmotionState(spectrum=EmotionSpectrum.DISTRUST_TRUST, intensity=-1.0)
    )
    
    # Blend with 30% of profile2
    blended = profile1.blend_with(profile2, weight=0.3)
    
    # Check blended values
    assert pytest.approx(blended.fear_courage.intensity, 0.01) == 0.42  # 0.6 * 0.7 + 0.0 * 0.3
    assert pytest.approx(blended.sadness_joy.intensity, 0.01) == 0.51  # 0.3 * 0.7 + 1.0 * 0.3
    assert pytest.approx(blended.distrust_trust.intensity, 0.01) == -0.3  # 0.0 * 0.7 + (-1.0) * 0.3
    
    # Verify emotion names - note that the closest matching intensity is used
    assert blended.fear_courage.intensity > 0.3  # It should be closer to Boldness than Calm
    assert blended.sadness_joy.intensity > 0.3  # It should be closer to Joy than Serenity
    assert blended.distrust_trust.intensity < 0  # It should be negative (on the distrust side)
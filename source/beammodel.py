#!/usr/bin/env python3
"""
BEAM (Behavioral Emotion Analysis Model) implementation using Pydantic v2.

This module provides a comprehensive implementation of the BEAM model,
which defines seven emotion spectra, each with varying intensity levels
from negative to positive. The model allows for quantification, representation,
and manipulation of emotional states.

Based on the research paper: "Behavioral Emotion Analysis Model for Large Language Models"
by Edward Y. Chang, Stanford University.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Dict, List, Literal, Optional, Union, Tuple
from enum import Enum
import numpy as np
from typing_extensions import Annotated


class EmotionSpectrum(str, Enum):
    """The seven emotion spectra defined in BEAM"""
    FEAR_COURAGE = "fear_courage"
    SADNESS_JOY = "sadness_joy"
    DISTRUST_TRUST = "distrust_trust"
    NEGLIGENCE_ANTICIPATION = "negligence_anticipation"
    ANGER_PEACE = "anger_peace"
    DISGUST_DELIGHT = "disgust_delight"
    DISINTEREST_FASCINATION = "disinterest_fascination"


# Standard intensity levels for emotions in BEAM
INTENSITY_VALUES = {
    -1.0: "extreme_negative", 
    -0.6: "moderate_negative",
    -0.3: "mild_negative",
    0.0: "neutral",
    0.3: "mild_positive",
    0.6: "moderate_positive",
    1.0: "extreme_positive"
}


# Define the emotions for each spectrum with their corresponding intensities
EMOTION_MAPPINGS = {
    EmotionSpectrum.FEAR_COURAGE: {
        -1.0: "Terror",
        -0.6: "Fear",
        -0.3: "Apprehension",
        0.0: "Calm",
        0.3: "Boldness",
        0.6: "Courage",
        1.0: "Heroism",
    },
    EmotionSpectrum.SADNESS_JOY: {
        -1.0: "Grief",
        -0.6: "Sadness",
        -0.3: "Pensiveness",
        0.0: "Surprise",
        0.3: "Serenity",
        0.6: "Joy",
        1.0: "Ecstasy",
    },
    EmotionSpectrum.DISTRUST_TRUST: {
        -1.0: "Distrust",
        -0.6: "Wary",
        -0.3: "Skepticism",
        0.0: "Acceptance",
        0.3: "Respect",
        0.6: "Trust",
        1.0: "Admiration",
    },
    EmotionSpectrum.NEGLIGENCE_ANTICIPATION: {
        -1.0: "Recklessness",
        -0.6: "Negligence",
        -0.3: "Apathy",
        0.0: "Cautiousness",
        0.3: "Interest",
        0.6: "Anticipation",
        1.0: "Vigilance",
    },
    EmotionSpectrum.ANGER_PEACE: {
        -1.0: "Rage",
        -0.6: "Anger",
        -0.3: "Annoyance",
        0.0: "Tolerance",
        0.3: "Composure",
        0.6: "Peace",
        1.0: "Tranquility",
    },
    EmotionSpectrum.DISGUST_DELIGHT: {
        -1.0: "Loathing",
        -0.6: "Disgust",
        -0.3: "Boredom",
        0.0: "Indifference",
        0.3: "Amusement",
        0.6: "Delight",
        1.0: "Enthusiasm",
    },
    EmotionSpectrum.DISINTEREST_FASCINATION: {
        -1.0: "Distraction",
        -0.6: "Disinterest",
        -0.3: "Unease",
        0.0: "Dullness",
        0.3: "Curiosity",
        0.6: "Fascination",
        1.0: "Amazement",
    },
}


class EmotionState(BaseModel):
    """Represents the state of a specific emotion within a spectrum"""
    spectrum: EmotionSpectrum
    intensity: Annotated[float, Field(ge=-1.0, le=1.0)]
    
    @property
    def emotion_name(self) -> str:
        """Get the emotion name based on intensity value"""
        # Find the closest predefined intensity level
        intensity_levels = list(EMOTION_MAPPINGS[self.spectrum].keys())
        closest_intensity = min(intensity_levels, key=lambda x: abs(x - self.intensity))
        return EMOTION_MAPPINGS[self.spectrum][closest_intensity]

    @classmethod
    def from_emotion_name(cls, spectrum: EmotionSpectrum, emotion_name: str) -> "EmotionState":
        """Create an EmotionState from a spectrum and emotion name"""
        # Find the intensity for the given emotion name
        for intensity, name in EMOTION_MAPPINGS[spectrum].items():
            if name.lower() == emotion_name.lower():
                return cls(spectrum=spectrum, intensity=float(intensity))
        
        # If not found, raise an error
        raise ValueError(f"Emotion '{emotion_name}' not found in spectrum {spectrum}")


class BEAMProfile(BaseModel):
    """Complete emotional profile using BEAM model"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    fear_courage: EmotionState = Field(
        default_factory=lambda: EmotionState(spectrum=EmotionSpectrum.FEAR_COURAGE, intensity=0.0)
    )
    sadness_joy: EmotionState = Field(
        default_factory=lambda: EmotionState(spectrum=EmotionSpectrum.SADNESS_JOY, intensity=0.0)
    )
    distrust_trust: EmotionState = Field(
        default_factory=lambda: EmotionState(spectrum=EmotionSpectrum.DISTRUST_TRUST, intensity=0.0)
    )
    negligence_anticipation: EmotionState = Field(
        default_factory=lambda: EmotionState(spectrum=EmotionSpectrum.NEGLIGENCE_ANTICIPATION, intensity=0.0)
    )
    anger_peace: EmotionState = Field(
        default_factory=lambda: EmotionState(spectrum=EmotionSpectrum.ANGER_PEACE, intensity=0.0)
    )
    disgust_delight: EmotionState = Field(
        default_factory=lambda: EmotionState(spectrum=EmotionSpectrum.DISGUST_DELIGHT, intensity=0.0)
    )
    disinterest_fascination: EmotionState = Field(
        default_factory=lambda: EmotionState(spectrum=EmotionSpectrum.DISINTEREST_FASCINATION, intensity=0.0)
    )
    
    @model_validator(mode='after')
    def validate_spectrum_assignments(self) -> "BEAMProfile":
        """Validate that each emotion state is assigned to the correct spectrum"""
        # Ensure each emotion state is assigned to its corresponding spectrum
        for field_name, expected_spectrum in [
            ("fear_courage", EmotionSpectrum.FEAR_COURAGE),
            ("sadness_joy", EmotionSpectrum.SADNESS_JOY),
            ("distrust_trust", EmotionSpectrum.DISTRUST_TRUST),
            ("negligence_anticipation", EmotionSpectrum.NEGLIGENCE_ANTICIPATION),
            ("anger_peace", EmotionSpectrum.ANGER_PEACE),
            ("disgust_delight", EmotionSpectrum.DISGUST_DELIGHT),
            ("disinterest_fascination", EmotionSpectrum.DISINTEREST_FASCINATION),
        ]:
            emotion_state = getattr(self, field_name)
            if emotion_state.spectrum != expected_spectrum:
                emotion_state.spectrum = expected_spectrum
        return self
    
    def get_dominant_emotions(self, n: int = 2) -> List[Dict]:
        """Get the n most dominant emotions (by absolute intensity)"""
        all_states = [
            self.fear_courage,
            self.sadness_joy,
            self.distrust_trust,
            self.negligence_anticipation,
            self.anger_peace,
            self.disgust_delight,
            self.disinterest_fascination
        ]
        
        # Sort by absolute intensity (descending)
        sorted_states = sorted(all_states, key=lambda s: abs(s.intensity), reverse=True)
        
        return [
            {
                "spectrum": state.spectrum.value,
                "emotion": state.emotion_name,
                "intensity": state.intensity
            }
            for state in sorted_states[:n]
        ]
    
    def to_vector(self) -> np.ndarray:
        """Convert the emotional profile to a 7-dimensional vector"""
        return np.array([
            self.fear_courage.intensity,
            self.sadness_joy.intensity,
            self.distrust_trust.intensity,
            self.negligence_anticipation.intensity,
            self.anger_peace.intensity,
            self.disgust_delight.intensity,
            self.disinterest_fascination.intensity
        ])
    
    @classmethod
    def from_vector(cls, vector: List[float]) -> "BEAMProfile":
        """Create a BEAMProfile from a 7-dimensional vector"""
        if len(vector) != 7:
            raise ValueError("Vector must have exactly 7 dimensions")
        
        return cls(
            fear_courage=EmotionState(spectrum=EmotionSpectrum.FEAR_COURAGE, intensity=vector[0]),
            sadness_joy=EmotionState(spectrum=EmotionSpectrum.SADNESS_JOY, intensity=vector[1]),
            distrust_trust=EmotionState(spectrum=EmotionSpectrum.DISTRUST_TRUST, intensity=vector[2]),
            negligence_anticipation=EmotionState(
                spectrum=EmotionSpectrum.NEGLIGENCE_ANTICIPATION, 
                intensity=vector[3]
            ),
            anger_peace=EmotionState(spectrum=EmotionSpectrum.ANGER_PEACE, intensity=vector[4]),
            disgust_delight=EmotionState(spectrum=EmotionSpectrum.DISGUST_DELIGHT, intensity=vector[5]),
            disinterest_fascination=EmotionState(
                spectrum=EmotionSpectrum.DISINTEREST_FASCINATION, 
                intensity=vector[6]
            )
        )
    
    @classmethod
    def create_from_emotion_names(cls, **emotions) -> "BEAMProfile":
        """
        Create a profile using emotion names
        Example: BEAMProfile.create_from_emotion_names(fear_courage="Courage", sadness_joy="Joy")
        """
        profile = cls()
        
        for field_name, emotion_name in emotions.items():
            if hasattr(profile, field_name):
                spectrum = getattr(profile, field_name).spectrum
                setattr(profile, field_name, EmotionState.from_emotion_name(spectrum, emotion_name))
        
        return profile
    
    def adjust_emotion(self, spectrum_name: str, adjustment: float) -> None:
        """
        Adjust an emotion's intensity by the specified amount, keeping within bounds
        
        Args:
            spectrum_name: Name of the emotion spectrum to adjust
            adjustment: Amount to adjust the intensity (-1.0 to 1.0)
        """
        if hasattr(self, spectrum_name):
            emotion_state = getattr(self, spectrum_name)
            new_intensity = min(max(emotion_state.intensity + adjustment, -1.0), 1.0)
            setattr(self, spectrum_name, EmotionState(
                spectrum=emotion_state.spectrum,
                intensity=new_intensity
            ))
    
    def blend_with(self, other_profile: "BEAMProfile", weight: float = 0.5) -> "BEAMProfile":
        """
        Blend this profile with another profile using the specified weight
        
        Args:
            other_profile: Another BEAMProfile to blend with
            weight: Weight of the other profile (0.0 to 1.0), where 0.0 is all this profile
                   and 1.0 is all other profile
        
        Returns:
            A new BEAMProfile that is a blend of the two profiles
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")
        
        this_vector = self.to_vector()
        other_vector = other_profile.to_vector()
        
        # Linear interpolation
        blended_vector = (1 - weight) * this_vector + weight * other_vector
        
        return BEAMProfile.from_vector(blended_vector.tolist())


# Example usage
if __name__ == "__main__":
    # Create a profile with default neutral values
    neutral_profile = BEAMProfile()
    print(f"Neutral profile: all emotions at intensity 0.0")
    
    # Create a profile with specific emotion intensities
    custom_profile = BEAMProfile(
        fear_courage=EmotionState(spectrum=EmotionSpectrum.FEAR_COURAGE, intensity=0.6),
        sadness_joy=EmotionState(spectrum=EmotionSpectrum.SADNESS_JOY, intensity=0.3),
        anger_peace=EmotionState(spectrum=EmotionSpectrum.ANGER_PEACE, intensity=-0.3)
    )
    print(f"\nCustom profile emotions:")
    print(f"Fear-Courage: {custom_profile.fear_courage.emotion_name}")
    print(f"Sadness-Joy: {custom_profile.sadness_joy.emotion_name}")
    print(f"Anger-Peace: {custom_profile.anger_peace.emotion_name}")
    
    # Create a profile from emotion names
    named_profile = BEAMProfile.create_from_emotion_names(
        fear_courage="Courage",
        sadness_joy="Ecstasy",
        distrust_trust="Distrust"
    )
    print(f"\nProfile created from emotion names:")
    print(f"Fear-Courage intensity: {named_profile.fear_courage.intensity}")
    print(f"Sadness-Joy intensity: {named_profile.sadness_joy.intensity}")
    print(f"Distrust-Trust intensity: {named_profile.distrust_trust.intensity}")
    
    # Get dominant emotions
    dominant = custom_profile.get_dominant_emotions(2)
    print(f"\nDominant emotions: {dominant}")
    
    # Convert to and from vector
    vector = custom_profile.to_vector()
    print(f"\nEmotion vector: {vector}")
    
    # Blend two profiles
    blended_profile = custom_profile.blend_with(named_profile, weight=0.3)
    print(f"\nBlended profile emotions:")
    print(f"Fear-Courage: {blended_profile.fear_courage.emotion_name}")
    print(f"Sadness-Joy: {blended_profile.sadness_joy.emotion_name}")
    print(f"Distrust-Trust: {blended_profile.distrust_trust.emotion_name}")
    print(f"Anger-Peace: {blended_profile.anger_peace.emotion_name}")
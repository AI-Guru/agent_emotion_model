from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator
from typing import Dict, List, Literal, Optional, Union, ClassVar, Tuple
from enum import Enum
import numpy as np
from typing_extensions import Annotated


class EmotionPair(str, Enum):
    """The four emotion pairs in Plutchik's model"""
    JOY_SADNESS = "joy_sadness"
    TRUST_DISGUST = "trust_disgust"
    FEAR_ANGER = "fear_anger"
    SURPRISE_ANTICIPATION = "surprise_anticipation"


class EmotionIntensity(str, Enum):
    """Intensity levels for emotions in Plutchik's wheel"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EmotionPolarity(str, Enum):
    """Polarity of the emotion (which side of the pair)"""
    POSITIVE = "positive"
    NEGATIVE = "negative"


# Complete emotion mapping with intensity levels
PLUTCHIK_EMOTION_MAP = {
    EmotionPair.JOY_SADNESS: {
        EmotionPolarity.POSITIVE: {
            EmotionIntensity.HIGH: "Ecstasy",
            EmotionIntensity.MEDIUM: "Joy",
            EmotionIntensity.LOW: "Serenity"
        },
        EmotionPolarity.NEGATIVE: {
            EmotionIntensity.HIGH: "Grief",
            EmotionIntensity.MEDIUM: "Sadness",
            EmotionIntensity.LOW: "Pensiveness"
        }
    },
    EmotionPair.TRUST_DISGUST: {
        EmotionPolarity.POSITIVE: {
            EmotionIntensity.HIGH: "Admiration",
            EmotionIntensity.MEDIUM: "Trust",
            EmotionIntensity.LOW: "Acceptance"
        },
        EmotionPolarity.NEGATIVE: {
            EmotionIntensity.HIGH: "Loathing",
            EmotionIntensity.MEDIUM: "Disgust",
            EmotionIntensity.LOW: "Boredom"
        }
    },
    EmotionPair.FEAR_ANGER: {
        EmotionPolarity.NEGATIVE: {  # Fear is considered negative
            EmotionIntensity.HIGH: "Terror",
            EmotionIntensity.MEDIUM: "Fear",
            EmotionIntensity.LOW: "Apprehension"
        },
        EmotionPolarity.POSITIVE: {  # Anger is considered positive in the sense of activation
            EmotionIntensity.HIGH: "Rage",
            EmotionIntensity.MEDIUM: "Anger",
            EmotionIntensity.LOW: "Annoyance"
        }
    },
    EmotionPair.SURPRISE_ANTICIPATION: {
        EmotionPolarity.NEGATIVE: {  # Surprise is considered negative in the sense of unexpectedness
            EmotionIntensity.HIGH: "Amazement",
            EmotionIntensity.MEDIUM: "Surprise",
            EmotionIntensity.LOW: "Distraction"
        },
        EmotionPolarity.POSITIVE: {  # Anticipation is considered positive
            EmotionIntensity.HIGH: "Vigilance",
            EmotionIntensity.MEDIUM: "Anticipation",
            EmotionIntensity.LOW: "Interest"
        }
    }
}


# Define compound emotions (dyads) based on primary emotions
PRIMARY_DYADS = {
    "Love": (EmotionPair.JOY_SADNESS, EmotionPolarity.POSITIVE, EmotionPair.TRUST_DISGUST, EmotionPolarity.POSITIVE),
    "Submission": (EmotionPair.TRUST_DISGUST, EmotionPolarity.POSITIVE, EmotionPair.FEAR_ANGER, EmotionPolarity.NEGATIVE),
    "Awe": (EmotionPair.FEAR_ANGER, EmotionPolarity.NEGATIVE, EmotionPair.SURPRISE_ANTICIPATION, EmotionPolarity.NEGATIVE),
    "Disapproval": (EmotionPair.SURPRISE_ANTICIPATION, EmotionPolarity.NEGATIVE, EmotionPair.JOY_SADNESS, EmotionPolarity.NEGATIVE),
    "Remorse": (EmotionPair.JOY_SADNESS, EmotionPolarity.NEGATIVE, EmotionPair.TRUST_DISGUST, EmotionPolarity.NEGATIVE),
    "Contempt": (EmotionPair.TRUST_DISGUST, EmotionPolarity.NEGATIVE, EmotionPair.FEAR_ANGER, EmotionPolarity.POSITIVE),
    "Aggressiveness": (EmotionPair.FEAR_ANGER, EmotionPolarity.POSITIVE, EmotionPair.SURPRISE_ANTICIPATION, EmotionPolarity.POSITIVE),
    "Optimism": (EmotionPair.SURPRISE_ANTICIPATION, EmotionPolarity.POSITIVE, EmotionPair.JOY_SADNESS, EmotionPolarity.POSITIVE),
}


class EmotionState(BaseModel):
    """Represents a specific emotion within Plutchik's model"""
    pair: EmotionPair
    polarity: EmotionPolarity
    intensity: EmotionIntensity
    
    @property
    def emotion_name(self) -> str:
        """Get the name of the emotion based on pair, polarity, and intensity"""
        return PLUTCHIK_EMOTION_MAP[self.pair][self.polarity][self.intensity]
    
    @classmethod
    def from_emotion_name(cls, emotion_name: str) -> "EmotionState":
        """Create an EmotionState from an emotion name"""
        # Search through the emotion map for the given name
        for pair in EmotionPair:
            for polarity in EmotionPolarity:
                for intensity in EmotionIntensity:
                    if PLUTCHIK_EMOTION_MAP[pair][polarity][intensity].lower() == emotion_name.lower():
                        return cls(pair=pair, polarity=polarity, intensity=intensity)
        
        raise ValueError(f"Emotion '{emotion_name}' not found in Plutchik's model")


class PlutchikProfile(BaseModel):
    """Represents a complete emotional profile using Plutchik's wheel of emotions"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # The eight basic emotions with their intensities
    joy: EmotionState = Field(
        default_factory=lambda: EmotionState(
            pair=EmotionPair.JOY_SADNESS,
            polarity=EmotionPolarity.POSITIVE,
            intensity=EmotionIntensity.MEDIUM
        )
    )
    sadness: EmotionState = Field(
        default_factory=lambda: EmotionState(
            pair=EmotionPair.JOY_SADNESS,
            polarity=EmotionPolarity.NEGATIVE,
            intensity=EmotionIntensity.MEDIUM
        )
    )
    trust: EmotionState = Field(
        default_factory=lambda: EmotionState(
            pair=EmotionPair.TRUST_DISGUST,
            polarity=EmotionPolarity.POSITIVE,
            intensity=EmotionIntensity.MEDIUM
        )
    )
    disgust: EmotionState = Field(
        default_factory=lambda: EmotionState(
            pair=EmotionPair.TRUST_DISGUST,
            polarity=EmotionPolarity.NEGATIVE,
            intensity=EmotionIntensity.MEDIUM
        )
    )
    fear: EmotionState = Field(
        default_factory=lambda: EmotionState(
            pair=EmotionPair.FEAR_ANGER,
            polarity=EmotionPolarity.NEGATIVE,
            intensity=EmotionIntensity.MEDIUM
        )
    )
    anger: EmotionState = Field(
        default_factory=lambda: EmotionState(
            pair=EmotionPair.FEAR_ANGER,
            polarity=EmotionPolarity.POSITIVE,
            intensity=EmotionIntensity.MEDIUM
        )
    )
    surprise: EmotionState = Field(
        default_factory=lambda: EmotionState(
            pair=EmotionPair.SURPRISE_ANTICIPATION,
            polarity=EmotionPolarity.NEGATIVE,
            intensity=EmotionIntensity.MEDIUM
        )
    )
    anticipation: EmotionState = Field(
        default_factory=lambda: EmotionState(
            pair=EmotionPair.SURPRISE_ANTICIPATION,
            polarity=EmotionPolarity.POSITIVE,
            intensity=EmotionIntensity.MEDIUM
        )
    )
    
    @model_validator(mode='after')
    def validate_emotion_assignments(self) -> "PlutchikProfile":
        """Validate that each emotion state is assigned correctly"""
        # Define expected assignments
        expected = {
            "joy": (EmotionPair.JOY_SADNESS, EmotionPolarity.POSITIVE),
            "sadness": (EmotionPair.JOY_SADNESS, EmotionPolarity.NEGATIVE),
            "trust": (EmotionPair.TRUST_DISGUST, EmotionPolarity.POSITIVE),
            "disgust": (EmotionPair.TRUST_DISGUST, EmotionPolarity.NEGATIVE),
            "fear": (EmotionPair.FEAR_ANGER, EmotionPolarity.NEGATIVE),
            "anger": (EmotionPair.FEAR_ANGER, EmotionPolarity.POSITIVE),
            "surprise": (EmotionPair.SURPRISE_ANTICIPATION, EmotionPolarity.NEGATIVE),
            "anticipation": (EmotionPair.SURPRISE_ANTICIPATION, EmotionPolarity.POSITIVE),
        }
        
        # Correct any misassignments
        for field_name, (expected_pair, expected_polarity) in expected.items():
            emotion_state = getattr(self, field_name)
            if emotion_state.pair != expected_pair or emotion_state.polarity != expected_polarity:
                emotion_state.pair = expected_pair
                emotion_state.polarity = expected_polarity
        
        return self
    
    def get_active_emotions(self, threshold: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM") -> List[Dict]:
        """
        Get all emotions at or above the specified intensity threshold
        Returns list of dicts with emotion names and their intensities
        """
        threshold_value = EmotionIntensity(threshold.lower())
        threshold_rank = list(EmotionIntensity).index(threshold_value)
        
        emotion_states = [
            self.joy, self.sadness, self.trust, self.disgust,
            self.fear, self.anger, self.surprise, self.anticipation
        ]
        
        # Filter emotions at or above threshold
        active_emotions = []
        for state in emotion_states:
            state_rank = list(EmotionIntensity).index(state.intensity)
            if state_rank >= threshold_rank:
                active_emotions.append({
                    "name": state.emotion_name,
                    "intensity": state.intensity
                })
        
        return active_emotions
    
    def get_dyads(self) -> Dict[str, float]:
        """
        Calculate the strength of compound emotions (dyads)
        Returns a dictionary of dyad names and their strength values
        """
        # Define a helper function to get intensity value
        def get_intensity_value(intensity: EmotionIntensity) -> float:
            if intensity == EmotionIntensity.HIGH:
                return 1.0
            elif intensity == EmotionIntensity.MEDIUM:
                return 0.5
            else:  # LOW
                return 0.2
        
        # Calculate strength for each primary dyad
        dyad_strengths = {}
        for dyad_name, (pair1, polarity1, pair2, polarity2) in PRIMARY_DYADS.items():
            # Find the corresponding emotions
            emotion1 = None
            emotion2 = None
            
            for emotion in [self.joy, self.sadness, self.trust, self.disgust, 
                           self.fear, self.anger, self.surprise, self.anticipation]:
                if emotion.pair == pair1 and emotion.polarity == polarity1:
                    emotion1 = emotion
                elif emotion.pair == pair2 and emotion.polarity == polarity2:
                    emotion2 = emotion
            
            if emotion1 and emotion2:
                # Dyad strength is the geometric mean of the two emotion intensities
                strength = (get_intensity_value(emotion1.intensity) * 
                            get_intensity_value(emotion2.intensity)) ** 0.5
                dyad_strengths[dyad_name] = strength
        
        return dyad_strengths
    
    def to_vector(self) -> np.ndarray:
        """
        Convert the emotional profile to a vector representation
        Returns an 8-dimensional vector with values representing emotion intensities
        """
        # Convert intensity to numeric value
        def intensity_to_value(intensity: EmotionIntensity) -> float:
            if intensity == EmotionIntensity.HIGH:
                return 1.0
            elif intensity == EmotionIntensity.MEDIUM:
                return 0.5
            else:  # LOW
                return 0.2
        
        return np.array([
            intensity_to_value(self.joy.intensity),
            intensity_to_value(self.sadness.intensity),
            intensity_to_value(self.trust.intensity),
            intensity_to_value(self.disgust.intensity),
            intensity_to_value(self.fear.intensity),
            intensity_to_value(self.anger.intensity),
            intensity_to_value(self.surprise.intensity),
            intensity_to_value(self.anticipation.intensity)
        ])
    
    @classmethod
    def from_vector(cls, vector: List[float]) -> "PlutchikProfile":
        """Create a PlutchikProfile from an 8-dimensional vector"""
        if len(vector) != 8:
            raise ValueError("Vector must have exactly 8 dimensions")
        
        # Convert numeric value to intensity
        def value_to_intensity(value: float) -> EmotionIntensity:
            if value >= 0.7:
                return EmotionIntensity.HIGH
            elif value >= 0.3:
                return EmotionIntensity.MEDIUM
            else:
                return EmotionIntensity.LOW
        
        return cls(
            joy=EmotionState(
                pair=EmotionPair.JOY_SADNESS,
                polarity=EmotionPolarity.POSITIVE,
                intensity=value_to_intensity(vector[0])
            ),
            sadness=EmotionState(
                pair=EmotionPair.JOY_SADNESS,
                polarity=EmotionPolarity.NEGATIVE,
                intensity=value_to_intensity(vector[1])
            ),
            trust=EmotionState(
                pair=EmotionPair.TRUST_DISGUST,
                polarity=EmotionPolarity.POSITIVE,
                intensity=value_to_intensity(vector[2])
            ),
            disgust=EmotionState(
                pair=EmotionPair.TRUST_DISGUST,
                polarity=EmotionPolarity.NEGATIVE,
                intensity=value_to_intensity(vector[3])
            ),
            fear=EmotionState(
                pair=EmotionPair.FEAR_ANGER,
                polarity=EmotionPolarity.NEGATIVE,
                intensity=value_to_intensity(vector[4])
            ),
            anger=EmotionState(
                pair=EmotionPair.FEAR_ANGER,
                polarity=EmotionPolarity.POSITIVE,
                intensity=value_to_intensity(vector[5])
            ),
            surprise=EmotionState(
                pair=EmotionPair.SURPRISE_ANTICIPATION,
                polarity=EmotionPolarity.NEGATIVE,
                intensity=value_to_intensity(vector[6])
            ),
            anticipation=EmotionState(
                pair=EmotionPair.SURPRISE_ANTICIPATION,
                polarity=EmotionPolarity.POSITIVE,
                intensity=value_to_intensity(vector[7])
            )
        )
    
    @classmethod
    def from_emotion_names(cls, **emotions) -> "PlutchikProfile":
        """
        Create a profile using specific emotion names
        Example: PlutchikProfile.from_emotion_names(joy="Ecstasy", fear="Terror")
        """
        profile = cls()
        
        # Map of field names to emotion attributes
        emotion_fields = {
            "joy": "joy",
            "sadness": "sadness",
            "trust": "trust",
            "disgust": "disgust",
            "fear": "fear",
            "anger": "anger",
            "surprise": "surprise",
            "anticipation": "anticipation"
        }
        
        # Update each specified emotion
        for field_name, emotion_name in emotions.items():
            if field_name in emotion_fields:
                try:
                    emotion_state = EmotionState.from_emotion_name(emotion_name)
                    setattr(profile, emotion_fields[field_name], emotion_state)
                except ValueError:
                    # If emotion name not found, skip this one
                    continue
        
        return profile


# Usage Example
if __name__ == "__main__":
    # Create a default profile with medium intensities
    default_profile = PlutchikProfile()
    print(f"Default profile emotions:")
    for emotion in ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]:
        state = getattr(default_profile, emotion)
        print(f"{emotion.capitalize()}: {state.emotion_name}")
    
    # Create a profile with specific intensities
    custom_profile = PlutchikProfile(
        joy=EmotionState(pair=EmotionPair.JOY_SADNESS, polarity=EmotionPolarity.POSITIVE, 
                        intensity=EmotionIntensity.HIGH),
        trust=EmotionState(pair=EmotionPair.TRUST_DISGUST, polarity=EmotionPolarity.POSITIVE,
                          intensity=EmotionIntensity.HIGH),
        fear=EmotionState(pair=EmotionPair.FEAR_ANGER, polarity=EmotionPolarity.NEGATIVE,
                         intensity=EmotionIntensity.LOW)
    )
    
    print("\nCustom profile active emotions (HIGH threshold):")
    print(custom_profile.get_active_emotions(threshold="HIGH"))
    
    # Get dyads (compound emotions)
    dyads = custom_profile.get_dyads()
    print("\nCompound emotions (dyads):")
    for dyad, strength in sorted(dyads.items(), key=lambda x: x[1], reverse=True):
        if strength > 0.3:  # Only show significant dyads
            print(f"{dyad}: {strength:.2f}")
    
    # Create from specific emotion names
    named_profile = PlutchikProfile.from_emotion_names(
        joy="Ecstasy",
        sadness="Pensiveness",
        fear="Terror",
        trust="Acceptance"
    )
    
    print("\nProfile created from emotion names:")
    print(f"Joy: {named_profile.joy.emotion_name} ({named_profile.joy.intensity})")
    print(f"Sadness: {named_profile.sadness.emotion_name} ({named_profile.sadness.intensity})")
    print(f"Fear: {named_profile.fear.emotion_name} ({named_profile.fear.intensity})")
    print(f"Trust: {named_profile.trust.emotion_name} ({named_profile.trust.intensity})")
    
    # Convert to and from vector
    vector = custom_profile.to_vector()
    print(f"\nEmotion vector: {vector}")
    
    reconstructed = PlutchikProfile.from_vector(vector)
    print(f"\nReconstructed profile emotions:")
    for emotion in ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]:
        state = getattr(reconstructed, emotion)
        print(f"{emotion.capitalize()}: {state.emotion_name}")
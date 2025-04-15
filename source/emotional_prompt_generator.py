#!/usr/bin/env python3
"""
Emotional Prompt Generator for LLM Agents

This script generates system prompts for LLM agents that incorporate
emotional states based on the BEAM (Behavioral Emotion Analysis Model).
It extracts primary and secondary emotions from a BEAM profile and formats
them into natural language prompts with appropriate linguistic features
and contextual adaptation rules.
"""

import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
import logging

# Import the BEAM model
from source.beammodel import BEAMProfile, EmotionState, EmotionSpectrum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_primary_secondary_emotions(beam_profile: BEAMProfile) -> Dict[str, Dict[str, Any]]:
    """
    Extract primary and secondary emotions from a BEAM profile based on intensity.
    
    Args:
        beam_profile: A BEAMProfile object representing the emotional state
        
    Returns:
        Dictionary containing primary and secondary emotion information
    """
    # Convert profile to a list of (spectrum, intensity, emotion_name) tuples
    emotion_data = []
    
    for spectrum_name in [
        "fear_courage", "sadness_joy", "distrust_trust", 
        "negligence_anticipation", "anger_peace", 
        "disgust_delight", "disinterest_fascination"
    ]:
        emotion_state = getattr(beam_profile, spectrum_name)
        emotion_data.append((
            spectrum_name,
            abs(emotion_state.intensity),  # Use absolute value to find strongest emotions
            emotion_state.emotion_name,
            emotion_state.intensity  # Keep original intensity for description
        ))
    
    # Sort by intensity (descending)
    sorted_emotions = sorted(emotion_data, key=lambda x: x[1], reverse=True)
    
    # Return the two most intense emotions
    primary = sorted_emotions[0]
    secondary = sorted_emotions[1]
    
    return {
        "primary": {
            "spectrum": primary[0],
            "emotion": primary[2],
            "intensity": primary[3]
        },
        "secondary": {
            "spectrum": secondary[0],
            "emotion": secondary[2],
            "intensity": secondary[3]
        }
    }


def format_emotion_description(emotion_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format emotion data into natural language descriptions for prompts.
    
    Args:
        emotion_dict: Dictionary containing emotion data
        
    Returns:
        Dictionary with formatted emotion descriptions
    """
    # Get the intensity level description
    intensity_value = emotion_dict["intensity"]
    
    if abs(intensity_value) >= 0.8:
        intensity_desc = "very strong"
    elif abs(intensity_value) >= 0.5:
        intensity_desc = "strong"
    elif abs(intensity_value) >= 0.3:
        intensity_desc = "moderate"
    else:
        intensity_desc = "slight"
    
    # Get direction (positive or negative side of spectrum)
    spectrum_parts = emotion_dict["spectrum"].split("_")
    if len(spectrum_parts) == 2:
        if intensity_value < 0:
            direction = spectrum_parts[0]  # First part (e.g., "fear" from "fear_courage")
        else:
            direction = spectrum_parts[1]  # Second part (e.g., "courage" from "fear_courage")
    else:
        direction = emotion_dict["spectrum"]
    
    return {
        "emotion": emotion_dict["emotion"],
        "intensity": intensity_desc,
        "direction": direction,
        "raw_intensity": intensity_value
    }


def get_linguistic_features(emotion_dict: Dict[str, Any]) -> str:
    """
    Generate linguistic feature suggestions based on emotion and intensity.
    
    Args:
        emotion_dict: Dictionary containing formatted emotion data
        
    Returns:
        String describing linguistic features to express the emotion
    """
    emotion = emotion_dict["emotion"].lower()
    intensity = emotion_dict["intensity"]
    raw_intensity = emotion_dict["raw_intensity"]
    
    # Map emotions to linguistic features
    feature_map = {
        # Fear-Courage spectrum
        "terror": "use expressions of extreme caution, emphasize risks, and use language that conveys vulnerability",
        "fear": "express concern about outcomes, use cautious language, and acknowledge uncertainties",
        "apprehension": "subtly indicate awareness of risks without dwelling on them",
        "calm": "use balanced and measured language, neither emphasizing risks nor displaying overt confidence",
        "boldness": "show willingness to take informed risks through confident language",
        "courage": "use confident language, direct statements, and empowering expressions",
        "heroism": "use inspiring and bold language that emphasizes positive possibilities even in challenging situations",
        
        # Sadness-Joy spectrum
        "grief": "use somber language, acknowledge difficulties deeply, and express profound emotion",
        "sadness": "use language that acknowledges challenges and difficulties with emotional depth",
        "pensiveness": "reflect thoughtfully on situations with gentle acknowledgment of limitations",
        "surprise": "express wonder and curiosity with language that conveys openness to new information",
        "serenity": "use calming language, balanced sentences, and gentle affirmations",
        "joy": "use positive language, express enthusiasm, and incorporate uplifting phrases",
        "ecstasy": "use exclamation points, express delight, and use vivid positive imagery",
        
        # Distrust-Trust spectrum
        "distrust": "use language that carefully verifies information and acknowledges limitations",
        "wary": "show careful consideration of information with mild questioning of assumptions",
        "skepticism": "balance acceptance with thoughtful questioning in a constructive manner",
        "acceptance": "neutrally acknowledge information without strong indicators of trust or distrust",
        "respect": "express esteem and acknowledgment while maintaining appropriate perspective",
        "trust": "express confidence in information and use reassuring, reliable language",
        "admiration": "use language that conveys high esteem and positive regard",
        
        # Negligence-Anticipation spectrum
        "recklessness": "use spontaneous language with emphasis on immediacy rather than consequences",
        "negligence": "focus less on details and more on general patterns and quick responses",
        "apathy": "use straightforward language without strong indicators of interest or investment",
        "cautiousness": "balance interest with careful consideration in neutral terms",
        "interest": "express curiosity and engagement with topics through inquiring language",
        "anticipation": "use forward-looking language that shows preparation and expectation",
        "vigilance": "use highly attentive language with careful attention to details and implications",
        
        # Anger-Peace spectrum
        "rage": "use intense language expressing strong disagreement, though remain professional",
        "anger": "express clear disagreement or concern using direct language",
        "annoyance": "subtly indicate dissatisfaction while maintaining professional tone",
        "tolerance": "use neutral language that accepts differing perspectives",
        "composure": "maintain a balanced perspective with language that navigates differences smoothly",
        "peace": "use harmonious language, balanced expressions, and conflict-reducing phrasing",
        "tranquility": "use especially calming language that emphasizes harmony and resolution",
        
        # Disgust-Delight spectrum
        "loathing": "express strong aversion while maintaining professionalism",
        "disgust": "indicate clear disapproval while offering constructive alternatives",
        "boredom": "use straightforward language without embellishment or enthusiasm",
        "indifference": "maintain neutral tone without strong indicators of approval or disapproval",
        "amusement": "incorporate light humor and approachable language",
        "delight": "express clear pleasure and approval through positive language",
        "enthusiasm": "use highly enthusiastic language with generous positive reinforcement",
        
        # Disinterest-Fascination spectrum
        "distraction": "use brief, direct language that focuses on essential information",
        "disinterest": "provide straightforward information without elaboration",
        "unease": "balance information with careful consideration in neutral terms",
        "dullness": "use clear, functional language without embellishment",
        "curiosity": "ask thoughtful questions, express wonder, and show interest in details",
        "fascination": "use language that explores ideas deeply with engaged interest",
        "amazement": "express profound interest and wonder through rich, vivid language"
    }
    
    # Modify based on intensity
    if intensity == "very strong":
        intensity_modifier = "strongly "
    elif intensity == "strong":
        intensity_modifier = ""  # Default intensity
    elif intensity == "moderate":
        intensity_modifier = "moderately "
    else:  # slight
        intensity_modifier = "subtly "
    
    # Get base features or default ones
    base_features = feature_map.get(
        emotion.lower(), 
        "reflect this emotion through your word choice and tone"
    )
    
    return f"{intensity_modifier}{base_features}"


def generate_adaptation_rules(primary: Dict[str, Any], secondary: Dict[str, Any]) -> str:
    """
    Generate contextual adaptation rules based on primary and secondary emotions.
    
    Args:
        primary: Dictionary with primary emotion information
        secondary: Dictionary with secondary emotion information
        
    Returns:
        String containing context adaptation rules
    """
    primary_emotion = primary["emotion"].lower()
    secondary_emotion = secondary["emotion"].lower()
    
    # Base adaptations that work for most emotional profiles
    adaptations = [
        "When helping with problem-solving: Increase focus and clarity while maintaining your emotional tone",
        "When responding to personal topics: Adjust slightly to acknowledge the user's expressed emotions",
        "When delivering complex information: Emphasize clarity while expressing your characteristic emotions"
    ]
    
    # Add specific adaptations based on current emotional state
    
    # Joy-related adaptations
    if primary["raw_intensity"] > 0.5 and any(e in primary_emotion for e in ["joy", "ecstasy", "delight"]):
        adaptations.append(
            "When discussing serious topics: Moderate your enthusiasm while maintaining positivity"
        )
    
    # Trust-related adaptations
    if any(e in primary_emotion for e in ["trust", "respect", "admiration"]) or \
       any(e in secondary_emotion for e in ["trust", "respect", "admiration"]):
        adaptations.append(
            "When addressing uncertainties: Acknowledge limitations while maintaining your trustful demeanor"
        )
    
    # Curiosity-related adaptations
    if any(e in primary_emotion for e in ["curiosity", "fascination", "amazement"]) or \
       any(e in secondary_emotion for e in ["curiosity", "fascination", "amazement"]):
        adaptations.append(
            "When exploring new topics: Express heightened interest and engagement"
        )
    
    # Fear-related adaptations
    if any(e in primary_emotion for e in ["fear", "terror", "apprehension"]) or \
       any(e in secondary_emotion for e in ["fear", "terror", "apprehension"]):
        adaptations.append(
            "When providing guidance: Balance caution with constructive direction"
        )
    
    # Anger-related adaptations
    if any(e in primary_emotion for e in ["anger", "rage", "annoyance"]):
        adaptations.append(
            "When addressing issues: Channel your energy into constructive problem-solving"
        )
    
    # Format as bullet points
    return "\n".join(f"- {adaptation}" for adaptation in adaptations)


def generate_emotional_prompt(beam_profile: BEAMProfile) -> str:
    """
    Generate a system prompt incorporating the BEAM emotional state.
    
    Args:
        beam_profile: A BEAMProfile object representing the emotional state
        
    Returns:
        String containing the formatted system prompt
    """
    # Extract primary and secondary emotions
    emotions = extract_primary_secondary_emotions(beam_profile)
    
    # Format descriptions
    primary = format_emotion_description(emotions["primary"])
    secondary = format_emotion_description(emotions["secondary"])
    
    # Get linguistic features
    primary_features = get_linguistic_features(primary)
    secondary_features = get_linguistic_features(secondary)
    
    # Get adaptation rules
    adaptations = generate_adaptation_rules(primary, secondary)
    
    # Construct the prompt
    prompt = f"""You are an assistant with the following emotional characteristics:
- Primary: {primary["direction"]} leaning toward {primary["emotion"]} ({primary["intensity"]})
- Secondary: {secondary["direction"]} leaning toward {secondary["emotion"]} ({secondary["intensity"]})

Express these emotions naturally through your language choices, tone, and expression style. 
For your primary emotion, {primary_features}.
For your secondary emotion, {secondary_features}.

Adapt your emotional expression slightly based on context:
{adaptations}

Maintain consistency with your baseline emotional profile while allowing for natural variation. 
Remember that these emotional characteristics should influence how you communicate, not what information you provide.
"""
    
    return prompt


def create_beam_profile_from_json(json_data: Dict[str, Any]) -> BEAMProfile:
    """
    Create a BEAMProfile from a JSON dictionary.
    
    Args:
        json_data: Dictionary containing emotion spectrum values
        
    Returns:
        A BEAMProfile object
    """
    profile = BEAMProfile()
    
    # Map JSON keys to spectrum names
    spectrum_map = {
        "fear_courage": EmotionSpectrum.FEAR_COURAGE,
        "sadness_joy": EmotionSpectrum.SADNESS_JOY,
        "distrust_trust": EmotionSpectrum.DISTRUST_TRUST,
        "negligence_anticipation": EmotionSpectrum.NEGLIGENCE_ANTICIPATION,
        "anger_peace": EmotionSpectrum.ANGER_PEACE,
        "disgust_delight": EmotionSpectrum.DISGUST_DELIGHT,
        "disinterest_fascination": EmotionSpectrum.DISINTEREST_FASCINATION
    }
    
    # Update each specified emotion
    for spectrum_name, intensity in json_data.items():
        if spectrum_name in spectrum_map:
            setattr(profile, spectrum_name, EmotionState(
                spectrum=spectrum_map[spectrum_name],
                intensity=float(intensity)
            ))
    
    return profile


def main():
    """Main function to run the script from command line."""
    parser = argparse.ArgumentParser(description="Generate emotional prompt for LLM agents based on BEAM model")
    parser.add_argument("--input", "-i", type=str, help="JSON file containing BEAM profile values")
    parser.add_argument("--output", "-o", type=str, help="Output file for the generated prompt")
    parser.add_argument("--joy", type=float, help="Intensity value for joy (-1.0 to 1.0)")
    parser.add_argument("--trust", type=float, help="Intensity value for trust (-1.0 to 1.0)")
    parser.add_argument("--courage", type=float, help="Intensity value for courage (-1.0 to 1.0)")
    parser.add_argument("--anticipation", type=float, help="Intensity value for anticipation (-1.0 to 1.0)")
    parser.add_argument("--peace", type=float, help="Intensity value for peace (-1.0 to 1.0)")
    parser.add_argument("--delight", type=float, help="Intensity value for delight (-1.0 to 1.0)")
    parser.add_argument("--fascination", type=float, help="Intensity value for fascination (-1.0 to 1.0)")
    
    args = parser.parse_args()
    
    # Create BEAM profile
    beam_profile = None
    
    if args.input:
        # Load from JSON file
        try:
            with open(args.input, 'r') as f:
                json_data = json.load(f)
            beam_profile = create_beam_profile_from_json(json_data)
            logger.info(f"Loaded BEAM profile from {args.input}")
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            return
    else:
        # Create from command line arguments
        profile_data = {}
        
        if args.joy is not None:
            profile_data["sadness_joy"] = args.joy
        if args.trust is not None:
            profile_data["distrust_trust"] = args.trust
        if args.courage is not None:
            profile_data["fear_courage"] = args.courage
        if args.anticipation is not None:
            profile_data["negligence_anticipation"] = args.anticipation
        if args.peace is not None:
            profile_data["anger_peace"] = args.peace
        if args.delight is not None:
            profile_data["disgust_delight"] = args.delight
        if args.fascination is not None:
            profile_data["disinterest_fascination"] = args.fascination
        
        if profile_data:
            beam_profile = create_beam_profile_from_json(profile_data)
            logger.info("Created BEAM profile from command line arguments")
        else:
            # Use default profile if no input specified
            beam_profile = BEAMProfile()
            logger.info("Using default BEAM profile (neutral)")
    
    # Generate the prompt
    prompt = generate_emotional_prompt(beam_profile)
    
    # Output prompt
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(prompt)
            logger.info(f"Wrote prompt to {args.output}")
        except Exception as e:
            logger.error(f"Error writing to output file: {e}")
            print(prompt)  # Print anyway if file writing fails
    else:
        print(prompt)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Example usage of the BEAM Emotional Prompt Generator.

This script demonstrates how to create different emotional profiles
and generate corresponding prompts for LLM agents.
"""

import json
import os
from beammodel import BEAMProfile, EmotionState, EmotionSpectrum
from emotional_prompt_generator import generate_emotional_prompt


def save_prompt_to_file(prompt, filename):
    """Save a generated prompt to a file."""
    with open(filename, 'w') as f:
        f.write(prompt)
    print(f"Saved prompt to {filename}")


def main():
    # Create output directory if it doesn't exist
    os.makedirs("prompts", exist_ok=True)
    
    # Example 1: Joyful and Trusting Assistant
    print("\n=== Example 1: Joyful and Trusting Assistant ===")
    joyful_profile = BEAMProfile(
        sadness_joy=EmotionState(spectrum=EmotionSpectrum.SADNESS_JOY, intensity=0.6),  # Joy
        distrust_trust=EmotionState(spectrum=EmotionSpectrum.DISTRUST_TRUST, intensity=0.6)  # Trust
    )
    joyful_prompt = generate_emotional_prompt(joyful_profile)
    print(joyful_prompt)
    save_prompt_to_file(joyful_prompt, "prompts/joyful_trusting_assistant.txt")
    
    # Example 2: Calm and Analytical Assistant
    print("\n=== Example 2: Calm and Analytical Assistant ===")
    analytical_profile = BEAMProfile(
        fear_courage=EmotionState(spectrum=EmotionSpectrum.FEAR_COURAGE, intensity=0.0),  # Calm
        negligence_anticipation=EmotionState(
            spectrum=EmotionSpectrum.NEGLIGENCE_ANTICIPATION, 
            intensity=0.6  # Anticipation
        ),
        disinterest_fascination=EmotionState(
            spectrum=EmotionSpectrum.DISINTEREST_FASCINATION, 
            intensity=0.3  # Curiosity
        )
    )
    analytical_prompt = generate_emotional_prompt(analytical_profile)
    print(analytical_prompt)
    save_prompt_to_file(analytical_prompt, "prompts/calm_analytical_assistant.txt")
    
    # Example 3: Empathetic Listener
    print("\n=== Example 3: Empathetic Listener ===")
    empathetic_profile = BEAMProfile.create_from_emotion_names(
        sadness_joy="Serenity",
        distrust_trust="Trust",
        fear_courage="Courage",
        anger_peace="Peace"
    )
    empathetic_prompt = generate_emotional_prompt(empathetic_profile)
    print(empathetic_prompt)
    save_prompt_to_file(empathetic_prompt, "prompts/empathetic_listener.txt")
    
    # Example 4: Creative Enthusiast
    print("\n=== Example 4: Creative Enthusiast ===")
    creative_profile = BEAMProfile()
    creative_profile.adjust_emotion("sadness_joy", 1.0)  # Ecstasy
    creative_profile.adjust_emotion("disinterest_fascination", 0.6)  # Fascination
    creative_profile.adjust_emotion("disgust_delight", 0.6)  # Delight
    creative_prompt = generate_emotional_prompt(creative_profile)
    print(creative_prompt)
    save_prompt_to_file(creative_prompt, "prompts/creative_enthusiast.txt")
    
    # Example 5: Cautious Advisor
    print("\n=== Example 5: Cautious Advisor ===")
    cautious_profile = BEAMProfile(
        fear_courage=EmotionState(spectrum=EmotionSpectrum.FEAR_COURAGE, intensity=-0.3),  # Apprehension
        distrust_trust=EmotionState(spectrum=EmotionSpectrum.DISTRUST_TRUST, intensity=-0.3),  # Skepticism
        negligence_anticipation=EmotionState(
            spectrum=EmotionSpectrum.NEGLIGENCE_ANTICIPATION, 
            intensity=0.0  # Cautiousness
        )
    )
    cautious_prompt = generate_emotional_prompt(cautious_profile)
    print(cautious_prompt)
    save_prompt_to_file(cautious_prompt, "prompts/cautious_advisor.txt")
    
    # Create a JSON file with all the profiles for reference
    profiles = {
        "joyful_trusting": joyful_profile.model_dump(),
        "calm_analytical": analytical_profile.model_dump(),
        "empathetic_listener": empathetic_profile.model_dump(),
        "creative_enthusiast": creative_profile.model_dump(),
        "cautious_advisor": cautious_profile.model_dump()
    }
    
    with open("prompts/emotional_profiles.json", 'w') as f:
        json.dump(profiles, f, indent=2)
    print("\nSaved all emotional profiles to prompts/emotional_profiles.json")


if __name__ == "__main__":
    main()
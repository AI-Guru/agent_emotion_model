# BEAM Emotion Integration for LLM Agents

This repository implements the Behavioral Emotion Analysis Model (BEAM) for Large Language Models as described in the paper by Edward Y. Chang. It provides a complete framework for modeling, managing, and integrating emotional states into LLM-powered agents.

## Overview

The BEAM framework offers a structured approach to modeling emotions in LLMs through seven distinct emotion spectra, each with varying intensities from negative to positive. This implementation provides:

1. A complete Pydantic v2 model of BEAM
2. Tools for generating emotion-aware system prompts
3. An `EmotionalAgent` class for managing emotional states
4. Example integrations with LLM APIs

## Components

### 1. BEAM Model (`source/beammodel.py`)

The core implementation of the BEAM framework as a Pydantic model, including:

- Seven emotion spectra (fear-courage, sadness-joy, distrust-trust, etc.)
- Quantified intensity levels (-1.0 to 1.0)
- Emotion state representation and manipulation
- Methods for blending emotional profiles

### 2. Emotional Prompt Generator (`source/emotional_prompt_generator.py`)

Transforms BEAM emotional states into natural language system prompts by:

- Extracting primary and secondary emotions from profiles
- Formatting appropriate linguistic expressions
- Generating contextual adaptation rules
- Producing complete system prompts

### 3. Emotional Agent (`source/emotional_agent.py`)

A class for managing LLM emotional states with features for:

- Dynamic emotional state updates
- Time-based emotional decay
- Adaptation to detected user emotions
- Interaction recording and state persistence

### 4. LLM Integration (`integration_with_llm.py`)

Example of integrating the emotional framework with OpenAI's API to create an interactive emotional chatbot.

## Usage

### Basic Usage

```python
# Import the components
from source.beammodel import BEAMProfile, EmotionState, EmotionSpectrum
from source.emotional_prompt_generator import generate_emotional_prompt

# Create an emotional profile
profile = BEAMProfile(
    sadness_joy=EmotionState(spectrum=EmotionSpectrum.SADNESS_JOY, intensity=0.6),  # Joy
    distrust_trust=EmotionState(spectrum=EmotionSpectrum.DISTRUST_TRUST, intensity=0.3)  # Respect
)

# Generate a system prompt with emotional guidelines
prompt = generate_emotional_prompt(profile)
print(prompt)
```

### Using the Emotional Agent

```python
from source.beammodel import BEAMProfile
from source.emotional_agent import EmotionalAgent

# Create an agent with an initial profile
agent = EmotionalAgent(
    initial_profile=profile,
    emotional_decay_rate=0.05,
    emotional_adaptation_rate=0.2,
    base_system_prompt="You are a helpful assistant."
)

# Update the emotional state based on user input
agent.update_emotional_state(
    emotion_adjustments={"fear_courage": 0.3}  # Increase courage
)

# Get the current system prompt
system_prompt = agent.get_system_prompt()
```

### Interactive Chat

Run the interactive chat example:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Run the interactive chat
python integration_with_llm.py --model gpt-3.5-turbo
```

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

The project uses pytest for testing. To run the tests:

```bash
python -m pytest
```

For more verbose output:

```bash
python -m pytest -v
```

## Examples

The repository includes several examples:

- `emotional_prompt_generator_demo.py`: Demonstrates creating emotional profiles and generating prompts
- `integration_with_llm.py`: Shows how to create an interactive chatbot with emotional awareness

## License

MIT

## Acknowledgments

Based on the research paper "Behavioral Emotion Analysis Model for Large Language Models" by Edward Y. Chang, Stanford University.
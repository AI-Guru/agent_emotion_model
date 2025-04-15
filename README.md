# BEAM Emotion Integration for LLM Agents

This repository implements the Behavioral Emotion Analysis Model (BEAM) for Large Language Models as described in the paper by Edward Y. Chang. It provides a complete framework for modeling, managing, and integrating emotional states into LLM-powered agents.

This implementation is also inspired by the paper ["An Emotion Model for Autonomous Agents"](https://arxiv.org/abs/2404.13071) by Tristan Behrens.

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
# Make sure you've set up your .env file with your OpenAI API key
# Then run the interactive chat
python integration_with_llm.py
```

## Installation

### Standard Installation

1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Copy `example.env` to `.env`
   - Replace the placeholder API key with your actual OpenAI API key
   ```bash
   cp example.env .env
   # Edit .env file with your API key
   ```

### Docker Installation

You can also run the application using Docker:

1. Clone the repository
2. Create your .env file with your OpenAI API key:
   ```bash
   cp example.env .env
   # Edit .env file with your API key
   ```
3. Build and run with docker-compose:
   ```bash
   docker-compose up -d
   ```
4. Access the application at http://localhost:7860 or http://[your-machine-ip]:7860 from other devices on your network

To stop the application:
```bash
docker-compose down
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
- `integration_with_llm.py`: Shows how to create an interactive command-line chatbot with emotional awareness
- `emotional_chatbot_app.py`: Provides a web interface for the emotional chatbot using Gradio

### Running the Web Interface

To run the Gradio web interface:

```bash
# Make sure you've set up your .env file with your OpenAI API key
python emotional_chatbot_app.py
```

To make the interface accessible from other devices on your network:

```bash
python emotional_chatbot_app.py --server-name 0.0.0.0
```

For additional options:

```bash
python emotional_chatbot_app.py --help
```

Command line options:
- `--model`: Specify the OpenAI model to use (default: gpt-4.1-mini-2025-04-14)
- `--temp`: Set the temperature for generation (default: 0.7)
- `--server-name`: Host to bind the server to (use 0.0.0.0 to make it accessible on your network)
- `--share`: Create a shareable link to access the app from another device
- `--decay`: Emotion decay factor for weighted history (default: 0.8)
- `--history`: Number of messages to include in emotional history (default: 5)

The web interface displays:
- A chat interface for conversing with the AI
- A graph showing the conversation's emotional profile across all spectra
- A summary of the dominant emotions detected in the conversation

#### Weighted Emotional Model

The web interface uses a weighted-history approach to emotions:
- Analyzes each user message for emotional content
- Maintains a history of detected emotions
- Applies a decay factor to give more weight to recent messages
- Creates a weighted average across the conversation history
- Results in more stable and contextually appropriate emotional responses

## License

MIT

## Acknowledgments

Based on the research paper "Behavioral Emotion Analysis Model for Large Language Models" by Edward Y. Chang, Stanford University.
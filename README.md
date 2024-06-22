# Sanskrit to English Translation App using DSPy and Ollama

This project implements a Sanskrit to English translation application using DSPy and Ollama, a local Large Language Model (LLM) framework.

## Overview

The `compile.py` script is the main entry point of the application. It performs the following key tasks:

1. Initializes an Ollama model for translation.
2. Loads translation examples from a JSON file.
3. Defines evaluation metrics for translation accuracy and similarity.
4. Uses DSPy's MIPRO to compile and optimize a translation module.
5. Saves the compiled translator model.
6. Demonstrates translation with a sample Sanskrit sentence.

## Key Components

- **TranslationModule**: Defines the core translation functionality.
- **CheckTranslation**: A signature for verifying translation accuracy.
- **MIPRO**: A DSPy teleprompter for optimizing the translation model.
- **Ollama**: Used as both the prompt model and task model for local LLM capabilities.

## Features

- Translates Sanskrit text to English.
- Uses few-shot learning with example translations.
- Optimizes the translation model using MIPRO.
- Evaluates translations based on accuracy and cosine similarity.

## Usage

To use the translator:

1. Ensure all dependencies are installed.
2. Run `compile.py` to train and compile the translation model.
3. The compiled model will be saved as 'translator-mipro.json'.
4. You can then use the compiled model for translating Sanskrit sentences to English.

## Example

The script includes an example translation of a Sanskrit sentence.

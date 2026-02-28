#!/bin/bash

echo "Pulling diverse open-source models for the AUEB Agent..."

# List of required diverse models to pull from Ollama
MODELS=(
    "llama3.2:latest"
    "deepseek-coder:1.3b"
    "qwen2.5-coder:7b"
    "llava:latest"
    "gemma2:2b"
    "qwen2.5:3b"
    "mistral:latest"
)

for model in "${MODELS[@]}"; do
    echo "Pulling $model..."
    ollama pull "$model"
done

echo "Done! You can now run verify_parallel.py with a truly diverse panel of experts."

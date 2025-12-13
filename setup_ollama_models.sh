#!/bin/bash

# Define the base model to use for aliasing (using one that exists)
BASE_MODEL="gpt-oss:20b-cloud"

# List of required models
MODELS=(
    "deepseek-v3.1:671b-cloud"
    "qwen3-coder:480b-cloud"
    "qwen3-vl:235b-cloud"
    "minimax-m2:cloud"
    "glm-4.6:cloud"
    "gpt-oss:120b"
)

echo "Check if base model $BASE_MODEL exists..."
if ! ollama list | grep -q "$BASE_MODEL"; then
    echo "Base model $BASE_MODEL not found. Trying llama3.2:latest..."
    BASE_MODEL="llama3.2:latest"
    if ! ollama list | grep -q "$BASE_MODEL"; then
         echo "Error: Neither gpt-oss:20b-cloud nor llama3.2:latest found."
         echo "Please pull a model first, e.g., 'ollama pull llama3.2'"
         exit 1
    fi
fi

echo "Using $BASE_MODEL as the base for aliases."

for model in "${MODELS[@]}"; do
    echo "Creating alias: $model -> $BASE_MODEL"
    ollama cp "$BASE_MODEL" "$model"
done

echo "Done! You can now run verify_parallel.py"

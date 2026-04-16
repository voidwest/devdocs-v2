#!/bin/bash

ollama serve &

sleep 5

echo "pulling model..."
ollama pull phi3:mini
echo "model pulled."

wait $!

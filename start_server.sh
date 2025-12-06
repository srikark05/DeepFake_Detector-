#!/bin/bash
# Quick start script for the DeepFake Detection Web App

echo "Starting DeepFake Detection Web Server..."
echo ""

# Check if model exists
if [ ! -f "trained_model.pkl" ]; then
    echo "⚠️  Model not found! Training model first..."
    echo "This may take a while..."
    python train_model.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Model training failed. Please check the error messages above."
        exit 1
    fi
fi

echo "✓ Model found. Starting server..."
echo ""
echo "🌐 Server will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""

python app.py


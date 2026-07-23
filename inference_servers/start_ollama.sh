#!/bin/bash
# Script khởi chạy Ollama Server cho LLM GGUF Qwen2.5-7B
echo "Khởi động Ollama Server..."
ollama serve &
sleep 5
echo "Nạp mô hình Qwen2.5 7B GGUF..."
ollama run qwen2.5:7b-instruct-q4_K_M ""

#!/bin/bash
# Script khởi chạy Triton Inference Server / ONNX Runtime Server cho các mô hình Embedding
echo "Khởi động Triton Inference Server..."
tritonserver --model-repository=/models --strict-model-config=false

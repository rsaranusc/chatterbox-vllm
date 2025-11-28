FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies using uv
RUN uv sync

# Create a directory for outputs
RUN mkdir -p /app/outputs

# Set environment variables for the application
ENV CHATTERBOX_CFG_SCALE=0.5
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Expose port for Gradio interface
EXPOSE 7860

# Default command
CMD ["python3", "example-tts.py"]
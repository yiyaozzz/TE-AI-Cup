# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive dialog during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev \
    tesseract-ocr \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install specific Python packages
RUN pip install boto3==1.34.88 \
    botocore==1.34.88 \
    doctr==1.9.0 \
    fitz==0.0.1.dev2 \
    fuzzywuzzy==0.18.0 \
    numpy==1.23.5 \
    opencv_contrib_python==4.9.0.80 \
    opencv_python==4.9.0.80 \
    pandas==1.5.3 \
    Pillow==10.3.0 \
    protobuf==5.26.1 \
    pytesseract==0.3.10 \
    python_doctr==0.8.1 \
    streamlit==1.32.2 \
    ultralytics==8.1.47

# Install Node.js using NVM and npm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash \
    && export NVM_DIR="$HOME/.nvm" \
    && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" \
    && nvm install v21.6.2 \
    && nvm use v21.6.2 \
    && nvm alias default v21.6.2

# Copy your application code to the container (assuming your code is in the current directory)
COPY . /app

# Install Node.js dependencies for SvelteKit
WORKDIR /app
RUN npm install

EXPOSE 8501 3000

CMD ["npm", "run", "dev"]

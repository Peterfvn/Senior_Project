FROM ubuntu:22.04

# Always update and install in one RUN statement
RUN apt-get update && apt-get install -y \
    apt-utils \
    software-properties-common \
    ca-certificates \
    curl \
    git \
    # python3.11 \
    # python3.11-venv \
    # python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

RUN git clone https://github.com/Peterfvn/Senior_Project.git /root/Senior_Project

WORKDIR /root/Senior_Project

# Just run git pull because I think it caches the code when im rebuilding the image
# This way I ensure the latest code is always used
RUN git pull

RUN pip install -r requirements.txt

RUN echo "NEURON_DATA_DIR=/root/Senior_Project/Neuron Data" > .env && \
    echo "VISUALIZATION_DIR=/root/Senior_Project/Visualizations" >> .env


CMD ["/bin/bash"]

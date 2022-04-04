# Training Models

## Quick Start

build docker image `docker build -t train docker`

start container with bind mounts `docker run --rm --gpus all -v $(pwd):/src -it train bash`

install dependencies inside container `pip install -r requirements.txt`

### Train dialogue generation models

`python -m train.nlg`

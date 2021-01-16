# PTS-clone-API
PTS filter clone API 
## Requirements
- docker 19.03 / 18.09
- GPU NVIDIA 
- ubuntu > 18.04
## Installation
### 1. Clone this repo
```bash
git clone https://github.com/tonhathuy/PTS-clone-API.git
```
### 2. Create docker image with dockerfile
```bash
cd PTS-clone-API
docker build -t <name_image>:v1 .
```
### 3. Run container from docker image 
```bash 
docker run -it --gpus all --name <name_containter> -v $(pwd):/backup/ -p network='host'  <name_image>:v1
```
- create screen for container
```bash 
docker exec -it <name_containter> bash
```
### 4. Run API in docker container
```bash
CUDA_VISIBLE_DEVICES=<num_nvidia> python3 app.py
```

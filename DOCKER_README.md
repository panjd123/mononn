# Docker Develope Quick Start 

```bash
git clone --recurse-submodules https://github.com/panjd123/mononn.git
```

## SSH Setup

Create `container_key/id_rsa`, `container_key/id_rsa.pub`, which will be container's key.

Create `dev_key/id_ed25519.pub` which is your own machine's public key.

## Build

```bash
docker build -t mononn_image2 .
```

It will build for a little bit long time.

```bash
screen -S mononn
docker build -t mononn_image2 .
```

## Run

```bash
docker run -it --name mononn2 \
    -p 25488:22 \
    --gpus all \
    --restart always \
    --privileged \
    mononn_image2
```

## Debug

```bash
docker run -it --name mononn2 \
    -p 25488:22 \
    --gpus all \
    --privileged \
    mononn_image2
```

```bash
docker stop mononn2
docker rm mononn2
```

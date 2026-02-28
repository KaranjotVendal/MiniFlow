# Docker Cheatsheet

Quick reference for common Docker commands and patterns.

## Building Images

```bash
# Build from Dockerfile
docker build -t myimage .

# Build with specific Dockerfile
docker build -t myimage -f Dockerfile.prod .

# Build without cache
docker build --no-cache -t myimage .

# Build with build args
docker build --build-arg VERSION=1.0 -t myimage .
```

## Running Containers

```bash
# Basic run
docker run myimage

# Run with port mapping
docker run -p 8000:8000 myimage

# Run with GPU
docker run --gpus all myimage

# Run with environment variables
docker run -e KEY=value -e ANOTHER=val myimage

# Run with volume
docker run -v /host/path:/container/path myimage

# Run in background (detached)
docker run -d myimage

# Run interactively
docker run -it myimage bash

# Run and remove after exit
docker run --rm myimage

# Run with resource limits
docker run --memory=8g --cpus=4 myimage

# Run with name
docker run --name mycontainer myimage
```

## Managing Containers

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop container
docker stop <container_id>

# Stop all containers
docker stop $(docker ps -q)

# Remove container
docker rm <container_id>

# Remove stopped containers
docker container prune

# View logs
docker logs <container_id>

# Follow logs
docker logs -f <container_id>

# Execute command in running container
docker exec -it <container_id> bash

# Copy files from container
docker cp <container_id>:/path/in/container /host/path

# Copy files to container
docker cp /host/path <container_id>:/path/in/container
```

## Managing Images

```bash
# List images
docker images

# Remove image
docker rmi myimage

# Remove dangling images
docker image prune

# Remove all unused images
docker image prune -a

# Tag image
docker tag myimage myimage:v1.0

# Push to registry
docker push myusername/myimage

# Pull from registry
docker pull myusername/myimage

# Save image to tar
docker save -o myimage.tar myimage

# Load image from tar
docker load -i myimage.tar
```

## Docker Compose

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild and start
docker-compose up --build

# View logs
docker-compose logs

# Scale service
docker-compose up -d --scale web=3
```

## System Maintenance

```bash
# View disk usage
docker system df

# Clean up unused data
docker system prune

# Clean up everything (volumes, networks, images)
docker system prune -a --volumes

# View Docker info
docker info

# View version
docker version
```

## GPU-Specific

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Run with specific GPU
docker run --gpus '"device=0"' myimage

# Run with multiple GPUs
docker run --gpus '"device=0,1,2"' myimage

# Run with GPU memory limit
docker run --gpus all --env NVIDIA_VISIBLE_DEVICES=0 myimage
```

## Debugging

```bash
# Inspect container
docker inspect <container_id>

# View container stats
docker stats

# View container processes
docker top <container_id>

# View image layers
docker history myimage

# Check container exit code
docker inspect <container_id> --format='{{.State.ExitCode}}'
```

## Useful One-Liners

```bash
# Remove all stopped containers
docker container prune -f

# Remove all unused images
docker image prune -af

# Remove all unused volumes
docker volume prune -f

# Remove everything
docker system prune -af --volumes

# Get container IP
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container_id>

# Get container environment variables
docker inspect -f '{{.Config.Env}}' <container_id>

# Show image size breakdown
docker history --no-trunc myimage | awk '{print $1, $7}'
```

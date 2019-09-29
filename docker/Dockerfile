FROM ubuntu

WORKDIR /home

# Avoiding user interaction with tzdata when installing
ENV DEBIAN_FRONTEND=noninteractive

# Update & Install packages
RUN apt-get update && apt-get install -y \
  python3-dev \
  python3-matplotlib \
  python3-scipy

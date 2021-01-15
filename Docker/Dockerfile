# Matt Lee (lees4@ornl.gov)
# Dongwon Shin (shind@ornl.gov)

FROM ubuntu:latest

EXPOSE 7777

# Update packages
RUN apt update && apt -y upgrade

# Install necessary packages
RUN apt install -y tar wget git curl bzip2 

# Download and install anaconda
RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
RUN bash Anaconda3-2020.11-Linux-x86_64.sh -b -p /anaconda

# Download ASCENDS
RUN git clone https://github.com/ornlpmcp/ASCENDS.git
RUN echo 'export PATH="/anaconda/bin:$PATH"' >> ~/.bashrc 
RUN echo "source activate ascends" > ~/.bashrc
ENV PATH /anaconda/bin:$PATH

# Create ascends conda environment
RUN /anaconda/bin/conda env create -f ASCENDS/environment.yml --name ascends 

# Launch ascends conda environment
SHELL ["/anaconda/bin/conda", "run", "-n", "ascends", "/bin/bash", "-c"]

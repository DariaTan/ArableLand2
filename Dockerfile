FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environments/environment.yml .
COPY . .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
RUN echo "conda activate al2" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure numpy is installed:"
RUN python -c "import numpy"

# The code to run when container is started:
# COPY run.py entrypoint.sh ./
# ENTRYPOINT ["./entrypoint.sh"]
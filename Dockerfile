FROM conda/miniconda3
RUN conda clean --all

# Install dependencies
RUN apt-get update && apt-get install -y wget\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
# Install requirements
RUN pip install -r requirements.txt
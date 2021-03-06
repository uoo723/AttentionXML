FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

WORKDIR /tmp
COPY requirements.txt requirements.txt

# RUN apt update && \
# 	apt install -y cifs-utils && \
# 	rm -rf /var/lib/apt/lists/*

RUN conda install -y python=3.8 conda setuptools && \
	conda install -y pytorch==1.7.0 cudatoolkit=10.2 -c pytorch && \
	conda install pillow==6.2.1 && \
	conda clean -a -y

RUN apt update && \
	apt install -y build-essential git && \
	rm -rf /var/lib/apt/lists/*

RUN pip install --ignore-installed -r requirements.txt --no-cache-dir

WORKDIR /workspace

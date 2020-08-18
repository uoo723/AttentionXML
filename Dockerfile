FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime
WORKDIR /tmp
COPY requirements.txt requirements.txt

# RUN apt update && \
# 	apt install -y cifs-utils && \
# 	rm -rf /var/lib/apt/lists/*

RUN conda install -y python=3.7 conda && \
	conda install -y pytorch=1.0.1 -c pytorch && \
	conda install pillow==6.2.1 && \
	conda clean -a -y

RUN pip install --ignore-installed -r requirements.txt --no-cache-dir

WORKDIR /workspace

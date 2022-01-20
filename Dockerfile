FROM nvcr.io/nvidia/pytorch:21.12-py3
# ARG OPENCV_VERSION=4.5.0
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update 

WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/
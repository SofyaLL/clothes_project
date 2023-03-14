FROM python:3.10.9

RUN python3 --version
RUN pip3 --version
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install opencv-python
RUN pip3 install absl-py numpy opencv-contrib-python protobuf
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install mediapipe

## Install bazel
#ARG BAZEL_VERSION=5.2.0
#RUN mkdir /bazel && \
#    wget --no-check-certificate -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/b\
#azel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
#    wget --no-check-certificate -O  /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
#    chmod +x /bazel/installer.sh && \
#    /bazel/installer.sh  && \
#    rm -f /bazel/installer.sh

# RUN pip3 install mediapipe

WORKDIR /project
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY project/ /project/
RUN ls -la /project/*
CMD ["python", "app.py"]
FROM triton_base:latest

WORKDIR /opt/tritonserver

RUN pip3 install pillow opencv-python-headless

COPY --chown=1000:1000 model_repository /opt/tritonserver/model_repository/

RUN	mkdir -p /models

ENV LD_PRELOAD=/opt/tritonserver/model_repository/libmmdeploy_tensorrt_ops.so

#cd server
#docker build docker/robotai/ -t triton_robotai:22.12.00

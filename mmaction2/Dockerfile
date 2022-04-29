# ARG PYTORCH="1.9.0"
# ARG CUDA="11.1"
# ARG CUDNN="8"

# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Install MMCV
# RUN pip install mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# Install MMDetection
RUN conda clean --all
# WORKDIR /home
RUN git clone https://github.com/vincent1rookie/10708-gcn_har.git 
# RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR 10708-gcn_har/mmaction2
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install -v -e .
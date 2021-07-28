FROM nvcr.io/nvidia/pytorch:21.06-py3

RUN pip install --upgrade pip && pip install \
    pytorch-lightning \
    timm \
    segmentation-models-pytorch

COPY ./main.py /workspace/main.py

WORKDIR /workspace
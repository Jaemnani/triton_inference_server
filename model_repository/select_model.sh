#!/bin/bash

echo "Select Models."
echo
echo "1. YOLOX-Tiny : Object Detection (COCO80 Classes) [Apache-2.0 license]"
echo "2. RTMDet-Ins-tiny : Instance Segmentation (COCO80 Classes) [Apache-2.0 license]"
echo

read -a models
#models=(1 2)

declare -A models_dict
models_dict["1"]="yolox_tiny_coco80"
models_dict["2"]="rtmdet_ins_tiny_coco80"

function install_yolox_tiny_coco80 {
    # yolox_tiny_preprocess
    echo " Install yolox_tiny_preprocess"
    mkdir -p /models/yolox_tiny_preprocess/1
    cp /opt/tritonserver/model_repository/yolox_tiny_preprocess/1/model.py /models/yolox_tiny_preprocess/1/model.py
    cp /opt/tritonserver/model_repository/yolox_tiny_preprocess/config.pbtxt /models/yolox_tiny_preprocess/config.pbtxt

    # yolox_tiny_postprocess
    echo " Install yolox_tiny_postprocess"
    mkdir -p /models/yolox_tiny_postprocess/1
    cp /opt/tritonserver/model_repository/yolox_tiny_postprocess/1/model.py /models/yolox_tiny_postprocess/1/model.py
    cp /opt/tritonserver/model_repository/yolox_tiny_postprocess/config.pbtxt /models/yolox_tiny_postprocess/config.pbtxt

    # ensemble_yolox_tiny_coco80
    echo " Install ensemble_yolox_tiny_coco80"
    mkdir -p /models/ensemble_yolox_tiny_coco80/1
    cp /opt/tritonserver/model_repository/ensemble_yolox_tiny_coco80/config.pbtxt /models/ensemble_yolox_tiny_coco80/config.pbtxt

    # yolox_tiny_coco80_trt_fp16
    echo " Install yolox_tiny_coco80_trt_fp16"
    mkdir -p /models/yolox_tiny_coco80_trt_fp16/1
    /usr/src/tensorrt/bin/trtexec --onnx=/opt/tritonserver/model_repository/yolox_tiny_coco80_trt_fp16/yolox_tiny_coco80.onnx --saveEngine=/opt/tritonserver/model_repository/yolox_tiny_coco80_trt_fp16/yolox_tiny_coco80_trt_fp16.plan --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
    cp /opt/tritonserver/model_repository/yolox_tiny_coco80_trt_fp16/yolox_tiny_coco80_trt_fp16.plan /models/yolox_tiny_coco80_trt_fp16/1/model.plan

    echo " Done"
}

function install_rtmdet_ins_tiny_coco80 {
    # rtmdet_ins_tiny_preprocess
    echo " Install rtmdet_ins_tiny_preprocess"
    mkdir -p /models/rtmdet_ins_tiny_preprocess/1
    cp /opt/tritonserver/model_repository/rtmdet_ins_tiny_preprocess/1/model.py /models/rtmdet_ins_tiny_preprocess/1/model.py
    cp /opt/tritonserver/model_repository/rtmdet_ins_tiny_preprocess/config.pbtxt /models/rtmdet_ins_tiny_preprocess/config.pbtxt

    # rtmdet_ins_tiny_postprocess
    echo " Install rtmdet_ins_tiny_postprocess"
    mkdir -p /models/rtmdet_ins_tiny_postprocess/1
    cp /opt/tritonserver/model_repository/rtmdet_ins_tiny_postprocess/1/model.py /models/rtmdet_ins_tiny_postprocess/1/model.py
    cp /opt/tritonserver/model_repository/rtmdet_ins_tiny_postprocess/config.pbtxt /models/rtmdet_ins_tiny_postprocess/config.pbtxt

    # ensemble_rtmdet_ins_tiny_coco80
    echo " Install ensemble_rtmdet_ins_tiny_coco80"
    mkdir -p /models/ensemble_rtmdet_ins_tiny_coco80/1
    cp /opt/tritonserver/model_repository/ensemble_rtmdet_ins_tiny_coco80/config.pbtxt /models/ensemble_rtmdet_ins_tiny_coco80/config.pbtxt

    # rtmdet_ins_tiny_coco80_trt_fp16
    echo " Install rtmdet_ins_tiny_coco80_trt_fp16"
    mkdir -p /models/rtmdet_ins_tiny_coco80_trt_fp16/1
    /usr/src/tensorrt/bin/trtexec --onnx=/opt/tritonserver/model_repository/rtmdet_ins_tiny_coco80_trt_fp16/rtmdet_ins_tiny_coco80.onnx --saveEngine=/opt/tritonserver/model_repository/rtmdet_ins_tiny_coco80_trt_fp16/rtmdet_ins_tiny_coco80_trt_fp16.plan --plugins=/opt/tritonserver/model_repository/libmmdeploy_tensorrt_ops.so --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw,int32:chw,fp16:chw --fp16
    cp /opt/tritonserver/model_repository/rtmdet_ins_tiny_coco80_trt_fp16/rtmdet_ins_tiny_coco80_trt_fp16.plan /models/rtmdet_ins_tiny_coco80_trt_fp16/1/model.plan

    echo " Done"
}

echo "Install the selected models:"
for num in "${models[@]}"
do
    echo -e "> ${models_dict[$num]}"
    if [ "${models_dict[$num]}" = "yolox_tiny_coco80" ]; then
        install_yolox_tiny_coco80
    elif [ "${models_dict[$num]}" = "rtmdet_ins_tiny_coco80" ]; then
        install_rtmdet_ins_tiny_coco80
    fi
done

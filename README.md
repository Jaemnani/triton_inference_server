# triton_inference_server
Triton Inference Server Example with Docker

## Install 
```
virtualenv venv_test
source venv_test/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

* 배포용 버전에 맞는 NVIDIA 공식 Server Repository 다운로드
* v2.29.0 = NGC Container 22.12
```
git clone -b v2.29.0 https://github.com/triton-inference-server/server.git
```

* 이미지 재구성 (기본 도커 이미지 -> 백앤드 포함 이미지로 재구성)
"triton_base"이름으로 도커 이미지 풀
```
cd server
python3 compose.py --output-name triton_base --backend tensorrt --backend python --backend onnxruntime --backend python --repoagent checksum --container-version 22.12
```

* 배포용 도커로 만들기
```
docker build -t triton_inference_server .
```

docker run -it --rm --name=triton_convert --gpus=all --ipc=host --pid=host --shm-size=1g -p 8000:8000 -p 8001:8001 -p 8002:8002 triton_inference_server

시동 후, 실행 -> 1, 2, 1 2 중 하나 선택하여 onnx를 tensorrt로 변환. (plan 확장자 파일 생성 됨.)
tensorrt모델이 환경 차이에 따라 다르게 변환되기 떄문에, 도커 이미지에 변환 하여 배포하면 동작하지 않을 수 있다.
반면에 Onnx는 모든 환경에서 똑같이 동작하기 떄문에, 미리 포함하여 배포,
사용될 서버 환경에서 변환을 하도록 하는 과정.

```
/opt/tritonserver/model_repository/select_model.sh
```

변환이 완료된 docker 컨테이너를 commit을 이용해 스냅샷 생성

```
docker commit -p triton_convert triton_model_server
```

이후는 최종 도커 이미지인 triton_model_server만 수행함
```
docker run -itd --name=triton_server --gpus=all --ipc=host --pid=host --shm-size=1g -p 8000:8000 -p 8001:8001 -p 8002:8002 triton_model_server tritonserver --model-repository=/models
```
* port
8000 : http
8001 : grpc infernce
8002 : metrics

사전 탑재된 모델
YOLOX-Tiny : Object Detection (COCO80 Classes) [Apache-2.0 license]
RTMDet-Ins-tiny : Instance Segmentation (COCO80 Classes) [Apache-2.0 license]

[참고] 디렉토리 구조 예시

파이썬 triton 모델은 아래와 같은 디렉토리 구조로 구성해야 함.


models
|-- model_a
|   |-- 1
|   |   `-- model.py
|   |-- config.pbtxt
|   |-- python3.6.tar.gz
|   `-- triton_python_backend_stub




[참고] 자료링크
https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/compose.md
https://github.com/triton-inference-server/python_backend
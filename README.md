# Triton Inference Server Example with Docker

This repository provides a guide and scripts to build a custom Triton Inference Server environment using Docker. It specifically addresses the need to convert ONNX models to TensorRT `.plan` files on the target GPU environment for maximum compatibility and performance.

## 🛠️ 1. Installation & Preparation

Set up the environment and download the necessary repositories.

```bash
# Create and activate virtual environment
virtualenv venv_test
source venv_test/bin/activate

# Install requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Download Official NVIDIA Server Repository
Clone the repository matching the version used for deployment.

* **Target Version:** v2.29.0 (Matches NGC Container 22.12)

```bash
git clone -b v2.29.0 [https://github.com/triton-inference-server/server.git](https://github.com/triton-inference-server/server.git)
```


## 🐳 2. Build Docker Images

### Step 1: Reconfigure Base Image

We need to create a custom base image because the default images might be too large or lack specific backends. We use `compose.py` to pull a "triton_base" image containing only the required backends (`tensorrt`, `python`, `onnxruntime`).

```Bash
cd server
python3 compose.py --output-name triton_base \
    --backend tensorrt \
    --backend python \
    --backend onnxruntime \
    --repoagent checksum \
    --container-version 22.12
```

### Step 2: Build Deployment Image

Build the final Docker image named `triton_inference_server`. This image will contain the model repository and dependencies.

```Bash
# Go back to the root directory
cd ..
docker build -t triton_inference_server .
```

---

## ⚙️ 3. Model Conversion Workflow (Important)

**Why this step is necessary:**
TensorRT models (`.plan` files) are highly optimized for specific GPU architectures. A plan file generated on one GPU (e.g., RTX 3090) might not work on another (e.g., T4 or A100).

- **Strategy:** Distribute **ONNX** models (universal) inside the Docker image.
- **Execution:** Convert ONNX to TensorRT **inside the container** on the actual server where it runs.

### Step 1: Run Conversion Container

Start a temporary container to perform the conversion.

```Bash
docker run -it --rm --name=triton_convert \
    --gpus=all --ipc=host --pid=host --shm-size=1g \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    triton_inference_server
```

### Step 2: Execute Conversion Script

Inside the running container, execute the `select_model.sh` script.

1. Select `1`, `2`, or `1 2` to convert the desired models.
2. This generates the `.plan` (TensorRT engine) files.

```Bash
# Run inside the container
/opt/tritonserver/model_repository/select_model.sh
```

### Step 3: Commit the Container

Once the conversion is finished and `.plan` files are created, save the container state as a new image (`triton_model_server`).

```Bash
# Run on the host machine
docker commit -p triton_convert triton_model_server
```

---

## 🚀 4. Final Execution

Run the final image (`triton_model_server`) which now contains the optimized TensorRT models.

```Bash
docker run -itd --name=triton_server \
    --gpus=all --ipc=host --pid=host --shm-size=1g \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    triton_model_server tritonserver --model-repository=/models
```

### Service Information

- **Ports:**
    - `8000`: HTTP
    - `8001`: gRPC Inference
    - `8002`: Metrics
- **Pre-loaded Models:**
    - **YOLOX-Tiny:** Object Detection (COCO 80 Classes) [Apache-2.0 license]
    - **RTMDet-Ins-tiny:** Instance Segmentation (COCO 80 Classes) [Apache-2.0 license]

---

## 📂 Reference

### Directory Structure Example

For models using the Python backend in Triton, the directory structure must follow this hierarchy:

```Text
models
|-- model_a
|   |-- 1
|   |   ㄴ-- model.py           # Python backend logic
|   |-- config.pbtxt           # Model configuration
|   |-- python3.6.tar.gz       # (Optional) Custom environment
|   ㄴ-- triton_python_backend_stub
```

### Useful Links

- [Triton Compose Guide](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/compose.md)
- [Triton Python Backend](https://github.com/triton-inference-server/python_backend)

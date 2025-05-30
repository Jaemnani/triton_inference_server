import json

import numpy as np
import triton_python_backend_utils as pb_utils

import cv2

class TritonPythonModel:
    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get output configuration
        output_config = pb_utils.get_output_config_by_name(model_config, "preprocess_output")

        # Convert Triton types to numpy types
        self.output_dtypes = pb_utils.triton_string_to_numpy(output_config["data_type"])
        
        params = model_config['parameters']
        self.model_input_size = tuple(map(int, self._get_params(params, "model_input_size").split(',')))

    def execute(self, requests):
        output_dtypes = self.output_dtypes
        model_input_size = self.model_input_size

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get input
            input_tensor = pb_utils.get_input_tensor_by_name(request, "preprocess_input")
            input_data = input_tensor.as_numpy()

            encoded_img = np.frombuffer(input_data[0], dtype = np.uint8)
            decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            img, ratio = self._preprocess(decoded_img, model_input_size)

            #output_data = img[None, :, :, :].astype(np.float16)
            #output_tensor = pb_utils.Tensor("preprocess_output", output_data.astype(output_dtypes))
            output_tensor = pb_utils.Tensor("preprocess_output", img.astype(output_dtypes))

            # Create InferenceResponse
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    output_tensor])
            responses.append(inference_response)
        return responses

    def finalize(self):
        pass
    
    def _get_params(self, params, keyname):
        for key, value in params.items():
            if key == keyname:
                parse_value = value["string_value"]
                break
        return parse_value

    def _preprocess(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * ratio), : int(img.shape[1] * ratio)] = resized_img

        # normalize image
        mean = np.array([103.53, 116.28, 123.675])
        std = np.array([57.375, 57.12, 58.395])
        padded_img = (padded_img - mean) / std

        padded_img = padded_img.transpose(swap)
        #padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float16)
        return padded_img, ratio

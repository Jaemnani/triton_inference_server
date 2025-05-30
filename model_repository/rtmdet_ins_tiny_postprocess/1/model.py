import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get output configuration
        output_boxes_config = pb_utils.get_output_config_by_name(model_config, "postprocess_output_boxes")
        output_scores_config = pb_utils.get_output_config_by_name(model_config, "postprocess_output_scores")
        output_classes_config = pb_utils.get_output_config_by_name(model_config, "postprocess_output_classes")
        output_masks_config = pb_utils.get_output_config_by_name(model_config, "postprocess_output_masks")

        # Convert Triton types to numpy types
        self.output_boxes_dtypes = pb_utils.triton_string_to_numpy(output_boxes_config["data_type"])
        self.output_scores_dtypes = pb_utils.triton_string_to_numpy(output_scores_config["data_type"])
        self.output_classes_dtypes = pb_utils.triton_string_to_numpy(output_classes_config["data_type"])
        self.output_masks_dtypes = pb_utils.triton_string_to_numpy(output_masks_config["data_type"])
        
        params = model_config['parameters']
        self.score_threshold = float(self._get_params(params, "score_threshold"))
        
    def execute(self, requests):
        output_boxes_dtypes = self.output_boxes_dtypes
        output_scores_dtypes = self.output_scores_dtypes
        output_classes_dtypes = self.output_classes_dtypes
        output_masks_dtypes = self.output_masks_dtypes

        score_threshold = self.score_threshold

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get post_input
            input_dets_tensor = pb_utils.get_input_tensor_by_name(request, "postprocess_input_dets")
            input_dets_data = input_dets_tensor.as_numpy()

            input_labels_tensor = pb_utils.get_input_tensor_by_name(request, "postprocess_input_labels")
            input_labels_data = input_labels_tensor.as_numpy()

            input_masks_tensor = pb_utils.get_input_tensor_by_name(request, "postprocess_input_masks")
            input_masks_data = input_masks_tensor.as_numpy()

            boxes, scores, cls_inds, masks = self._postprocess(input_dets_data, input_labels_data, input_masks_data, 
                                                               score_thr=score_threshold)

            output_boxes_tensor = pb_utils.Tensor("postprocess_output_boxes", boxes.astype(output_boxes_dtypes))
            output_scores_tensor = pb_utils.Tensor("postprocess_output_scores", scores.astype(output_scores_dtypes))
            output_classes_tensor = pb_utils.Tensor("postprocess_output_classes", cls_inds.astype(output_classes_dtypes))
            output_masks_tensor = pb_utils.Tensor("postprocess_output_masks", masks.astype(output_masks_dtypes))

            # Create InferenceResponse
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    output_boxes_tensor,
                    output_scores_tensor,
                    output_classes_tensor,
                    output_masks_tensor])
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

    def _postprocess(self, dets_data, labels_data, masks_data, score_thr=0.1):
        # onnx end2end
        if masks_data is not None:
            boxes, scores = dets_data[:, :4], dets_data[:, 4]
            cls_inds = labels_data
            masks = masks_data
            valid_score_mask = scores > score_thr
            valid_boxes = boxes[valid_score_mask]
            valid_scores = scores[valid_score_mask]
            valid_cls_inds = cls_inds[valid_score_mask]
            valid_masks = masks[valid_score_mask]
        else:
            valid_boxes = np.array([])
            valid_scores = np.array([])
            valid_cls_inds = np.array([])
            valid_masks = np.array([])

        return valid_boxes, valid_scores, valid_cls_inds, valid_masks

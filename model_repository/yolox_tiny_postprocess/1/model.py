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

        # Convert Triton types to numpy types
        self.output_boxes_dtypes = pb_utils.triton_string_to_numpy(output_boxes_config["data_type"])
        self.output_scores_dtypes = pb_utils.triton_string_to_numpy(output_scores_config["data_type"])
        self.output_classes_dtypes = pb_utils.triton_string_to_numpy(output_classes_config["data_type"])
        
        params = model_config['parameters']
        self.score_threshold = float(self._get_params(params, "score_threshold"))
        self.nms_threshold = float(self._get_params(params, "nms_threshold"))
        self.model_input_size = tuple(map(int, self._get_params(params, "model_input_size").split(',')))
        
    def execute(self, requests):
        output_boxes_dtypes = self.output_boxes_dtypes
        output_scores_dtypes = self.output_scores_dtypes
        output_classes_dtypes = self.output_classes_dtypes

        score_threshold = self.score_threshold
        nms_threshold = self.nms_threshold
        model_input_size = self.model_input_size

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get post_input
            input_tensor = pb_utils.get_input_tensor_by_name(request, "postprocess_input")
            input_data = input_tensor.as_numpy()

            input_data_copy = np.copy(input_data)
            input_data_copy = input_data_copy.astype(np.float32)
            predictions = self._postprocess(input_data_copy, model_input_size)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.

            dets = self._multiclass_nms(boxes_xyxy, scores, nms_thr=nms_threshold, score_thr=score_threshold)

            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            else:
                final_boxes = np.array([])
                final_scores = np.array([])
                final_cls_inds = np.array([])

            output_boxes_tensor = pb_utils.Tensor("postprocess_output_boxes", final_boxes.astype(output_boxes_dtypes))
            output_scores_tensor = pb_utils.Tensor("postprocess_output_scores", final_scores.astype(output_scores_dtypes))
            output_classes_tensor = pb_utils.Tensor("postprocess_output_classes", final_cls_inds.astype(output_classes_dtypes))

            # Create InferenceResponse
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    output_boxes_tensor,
                    output_scores_tensor,
                    output_classes_tensor])
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

    def _nms(self, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def _multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy"""
        nms_method = self._multiclass_nms_class_agnostic

        return nms_method(boxes, scores, nms_thr, score_thr)

    def _multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self._nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets

    def _postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

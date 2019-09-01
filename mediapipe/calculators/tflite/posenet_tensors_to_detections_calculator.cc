// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <unordered_map>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/tflite/posenet_tensors_to_detections_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "tensorflow/lite/interpreter.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"

#if defined(__ANDROID__)
#include <android/log.h>
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif  // ANDROID

#if defined(__ANDROID__)
using ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer;
using ::tflite::gpu::gl::GlBuffer;
using ::tflite::gpu::gl::GlProgram;
using ::tflite::gpu::gl::GlShader;
#endif  // ANDROID

namespace mediapipe {

namespace {

constexpr int kNumInputTensorsWithAnchors = 3;
constexpr int kNumCoordsPerBox = 4;

void ConvertRawValuesToAnchors(const float* raw_anchors, int num_boxes,
                               std::vector<Anchor>* anchors) {
  anchors->clear();
  for (int i = 0; i < num_boxes; ++i) {
    Anchor new_anchor;
    new_anchor.set_y_center(raw_anchors[i * kNumCoordsPerBox + 0]);
    new_anchor.set_x_center(raw_anchors[i * kNumCoordsPerBox + 1]);
    new_anchor.set_h(raw_anchors[i * kNumCoordsPerBox + 2]);
    new_anchor.set_w(raw_anchors[i * kNumCoordsPerBox + 3]);
    anchors->push_back(new_anchor);
  }
}

void ConvertAnchorsToRawValues(const std::vector<Anchor>& anchors,
                               int num_boxes, float* raw_anchors) {
  CHECK_EQ(anchors.size(), num_boxes);
  int box = 0;
  for (auto anchor : anchors) {
    raw_anchors[box * kNumCoordsPerBox + 0] = anchor.y_center();
    raw_anchors[box * kNumCoordsPerBox + 1] = anchor.x_center();
    raw_anchors[box * kNumCoordsPerBox + 2] = anchor.h();
    raw_anchors[box * kNumCoordsPerBox + 3] = anchor.w();
    ++box;
  }
}

const int WIDTH_FULL = 257;
const int HEIGHT_FULL = 353;
const int WIDTH_SMALL = 33;
const int HEIGHT_SMALL = 45;
const int NUM_KEYPOINTS = 17;
const int NUM_CLASSES = 24;

template <class T>
class View {
 public:
  View(const T* data, int h, int w, int c) : data_(data), h_(h), w_(w), c_(c), w_stride_(c), h_stride_(c * w) {}
  std::size_t index(int i, int j, int k) const {
    return h_stride_ * i + w_stride_ * j + k;
  }
  const T& operator()(int i, int j, int k) const {
    return data_[index(i, j, k)];
  }
  const T* data_;
  int h_, w_, c_;
  int w_stride_, h_stride_;
};

struct Keypoint {
  float y;
  float x;
  float confidence;
};
}  // namespace

// Convert result TFLite tensors from object detection models into MediaPipe
// Detections.
//
// Input:
//  TENSORS - Vector of TfLiteTensor of type kTfLiteFloat32. The vector of
//               tensors can have 2 or 3 tensors. First tensor is the predicted
//               raw boxes/keypoints. The size of the values must be (num_boxes
//               * num_predicted_values). Second tensor is the score tensor. The
//               size of the valuse must be (num_boxes * num_classes). It's
//               optional to pass in a third tensor for anchors (e.g. for SSD
//               models) depend on the outputs of the detection model. The size
//               of anchor tensor must be (num_boxes * 4).
//  TENSORS_GPU - vector of GlBuffer.
// Output:
//  DETECTIONS - Result MediaPipe detections.
//
// Usage example:
// node {
//   calculator: "PoseNetTensorsToDetectionsCalculator"
//   input_stream: "TENSORS:tensors"
//   input_side_packet: "ANCHORS:anchors"
//   output_stream: "DETECTIONS:detections"
//   options: {
//     [mediapipe.PoseNetTensorsToDetectionsCalculatorOptions.ext] {
//       num_classes: 91
//       num_boxes: 1917
//       num_coords: 4
//       ignore_classes: [0, 1, 2]
//       x_scale: 10.0
//       y_scale: 10.0
//       h_scale: 5.0
//       w_scale: 5.0
//     }
//   }
// }
class PoseNetTensorsToDetectionsCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  ::mediapipe::Status ProcessCPU(CalculatorContext* cc,
                                 std::vector<Detection>* output_detections);
  ::mediapipe::Status ProcessGPU(CalculatorContext* cc,
                                 std::vector<Detection>* output_detections);

  ::mediapipe::Status LoadOptions(CalculatorContext* cc);
  ::mediapipe::Status GlSetup(CalculatorContext* cc);
  ::mediapipe::Status DecodeBoxes(const float* raw_boxes,
                                  const std::vector<Anchor>& anchors,
                                  std::vector<float>* boxes);
  ::mediapipe::Status ConvertToDetections(
      const float* detection_boxes, const float* detection_scores,
      const int* detection_classes, std::vector<Detection>* output_detections);
  Detection ConvertToDetection(float box_ymin, float box_xmin, float box_ymax,
                               float box_xmax, float score, int class_id,
                               bool flip_vertically);

  int num_classes_ = 0;
  int num_boxes_ = 0;
  int num_coords_ = 0;
  std::set<int> ignore_classes_;

  ::mediapipe::PoseNetTensorsToDetectionsCalculatorOptions options_;
  std::vector<Anchor> anchors_;
  bool side_packet_anchors_{};

#if defined(__ANDROID__)
  mediapipe::GlCalculatorHelper gpu_helper_;
  std::unique_ptr<GlProgram> decode_program_;
  std::unique_ptr<GlProgram> score_program_;
  std::unique_ptr<GlBuffer> decoded_boxes_buffer_;
  std::unique_ptr<GlBuffer> raw_boxes_buffer_;
  std::unique_ptr<GlBuffer> raw_anchors_buffer_;
  std::unique_ptr<GlBuffer> scored_boxes_buffer_;
  std::unique_ptr<GlBuffer> raw_scores_buffer_;
#endif

  bool gpu_input_ = false;
  bool anchors_init_ = false;
};
REGISTER_CALCULATOR(PoseNetTensorsToDetectionsCalculator);

::mediapipe::Status PoseNetTensorsToDetectionsCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  if (cc->Inputs().HasTag("TENSORS")) {
    cc->Inputs().Tag("TENSORS").Set<std::vector<TfLiteTensor>>();
  }

#if defined(__ANDROID__)
  if (cc->Inputs().HasTag("TENSORS_GPU")) {
    cc->Inputs().Tag("TENSORS_GPU").Set<std::vector<GlBuffer>>();
  }
#endif

  //if (cc->Outputs().HasTag("DETECTIONS")) {
  //  cc->Outputs().Tag("DETECTIONS").Set<std::vector<Detection>>();
  //}

  if (cc->Inputs().HasTag("VIZ")) {
    cc->Inputs().Tag("VIZ").Set<ImageFrame>();
  }
  if (cc->Outputs().HasTag("VIZ")) {
    cc->Outputs().Tag("VIZ").Set<ImageFrame>();
  }

  if (cc->InputSidePackets().UsesTags()) {
    if (cc->InputSidePackets().HasTag("ANCHORS")) {
      cc->InputSidePackets().Tag("ANCHORS").Set<std::vector<Anchor>>();
    }
  }

#if defined(__ANDROID__)
  RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PoseNetTensorsToDetectionsCalculator::Open(
    CalculatorContext* cc) {
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "tensor_to_detection open 000");
  cc->SetOffset(TimestampDiff(0));

  if (cc->Inputs().HasTag("TENSORS_GPU")) {
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "tensor_to_detection open 001");
    gpu_input_ = true;
#if defined(__ANDROID__)
    RETURN_IF_ERROR(gpu_helper_.Open(cc));
#endif
  }

  // RETURN_IF_ERROR(LoadOptions(cc));
  __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "tensor_to_detection open 002");
  side_packet_anchors_ = cc->InputSidePackets().HasTag("ANCHORS");
  __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "tensor_to_detection open 003");

  if (gpu_input_) {
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "tensor_to_detection open 004");
    RETURN_IF_ERROR(GlSetup(cc));
  }

  __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "tensor_to_detection open 005");
  return ::mediapipe::OkStatus();
}

::mediapipe::Status PoseNetTensorsToDetectionsCalculator::Process(
    CalculatorContext* cc) {
  __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "tensor_to_detection process 000");
  if ((!gpu_input_ && cc->Inputs().Tag("TENSORS").IsEmpty()) ||
      (gpu_input_ && cc->Inputs().Tag("TENSORS_GPU").IsEmpty())) {
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "tensor_to_detection process 001");
    return ::mediapipe::OkStatus();
  }

  auto output_detections = absl::make_unique<std::vector<Detection>>();

  if (gpu_input_) {
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "tensor_to_detection process 002");
    RETURN_IF_ERROR(ProcessGPU(cc, output_detections.get()));
  } else {
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "tensor_to_detection process 003");
    RETURN_IF_ERROR(ProcessCPU(cc, output_detections.get()));
  }  // if gpu_input_

  // Output
  //if (cc->Outputs().HasTag("DETECTIONS")) {
  //  cc->Outputs()
  //      .Tag("DETECTIONS")
  //      .Add(output_detections.release(), cc->InputTimestamp());
  //}

  return ::mediapipe::OkStatus();
}

float lerp(const float v0, const float v1, const float t) {
  return (1 - t) * v0 + t * v1;
} 

float BilinearSample(
    const float* img, const float x, const float y, const float z, const int w, const int h, const int d) {
  const int y0 = static_cast<int>(y);
  const int x0 = static_cast<int>(x);
  const int y1 = std::min((int)(y0 + 1), h - 1);
  const int x1 = std::min((int)(x0 + 1), w - 1);
  
  const float yfrac = y - y0;
  const float xfrac = x - x0;
  
  const int idx00 = d * (y0 * w + x0) + z;
  const int idx01 = d * (y0 * w + x1) + z;
  const int idx10 = d * (y1 * w + x0) + z;
  const int idx11 = d * (y1 * w + x1) + z;

  const float r0 = lerp(img[idx00], img[idx01], xfrac);
  const float r1 = lerp(img[idx10], img[idx11], xfrac);

  return lerp(r0, r1, yfrac);
}

void DecodeSegmentation(const float* part_heatmap, const float* raw_segments, unsigned char* decoded) {
  const int image_size_y = HEIGHT_FULL;
  const int image_size_x = WIDTH_FULL;
  const int model_size_y = HEIGHT_SMALL;
  const int model_size_x = WIDTH_SMALL;

  for (int row = 0; row < image_size_y; ++row) {
    for (int col = 0; col < image_size_x; ++col) {
      const float y = (row + 0.5) / image_size_y * model_size_y;
      const float x = (col + 0.5) / image_size_x * model_size_x;
      const float segment_val = 
          BilinearSample(raw_segments, x, y, 0, model_size_x, model_size_y, 1);
      *decoded = 0;
      if (segment_val >= -0.4f) {
        float curr_max = -1.e7;
        int curr_max_index = -1;
        for (int part = 0; part < NUM_CLASSES; ++part) {
          const float val = BilinearSample(part_heatmap, x, y, part, model_size_x, model_size_y, NUM_CLASSES);
          if (val > curr_max) {
            curr_max = val;
            curr_max_index = part;
          }
        }
        *decoded = curr_max_index + 1;
      }
      decoded++;
    }

  }
}

void DecodePose(const float* point_heatmaps, const float* point_offsets, Keypoint* output_coords) {
  float maxvals[NUM_KEYPOINTS] = {};
  for (int k = 0; k < NUM_KEYPOINTS; ++k) {
    maxvals[k] = -1e7f;
  }
  int coords[NUM_KEYPOINTS][2] = {};
  const float* heat = point_heatmaps;
  for (int i = 0; i < HEIGHT_SMALL; ++i) {
    for (int j = 0; j < WIDTH_SMALL; ++j) {
      for (int k = 0; k < NUM_KEYPOINTS; ++k) {
        if (*heat > maxvals[k]) {
          maxvals[k] = *heat;
          coords[k][0] = i;
          coords[k][0] = j;
        }
        heat++;
      }
    }
  }

  View<float> offset_view(point_offsets, HEIGHT_SMALL, WIDTH_SMALL, NUM_KEYPOINTS);
  Keypoint* out = output_coords;
  for (int keypoint = 0; keypoint < NUM_KEYPOINTS; ++keypoint) {
    int i = coords[keypoint][0];
    int j = coords[keypoint][1];
    const float y_offset = offset_view(i, j, keypoint);
    const float x_offset = offset_view(i, j, NUM_KEYPOINTS + keypoint);
    float final_y = i / (HEIGHT_SMALL - 1.f) * HEIGHT_FULL + y_offset;
    float final_x = j / (WIDTH_SMALL - 1.f) * WIDTH_FULL + x_offset;
    out->y = final_y;
    out->x = final_x;
    out->confidence = maxvals[keypoint];
    out++;
  }
}

::mediapipe::Status PoseNetTensorsToDetectionsCalculator::ProcessCPU(
    CalculatorContext* cc, std::vector<Detection>* output_detections) {
  const auto& input_tensors =
      cc->Inputs().Tag("TENSORS").Get<std::vector<TfLiteTensor>>();

  __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", 
    "input_tensors.size() %lu ", input_tensors.size());  
  if (input_tensors.size() == 2) {
    // Postprocessing on CPU for model without postprocessing op. E.g. output
    // raw score tensor and box tensor. Anchor decoding will be handled below.
    const TfLiteTensor* raw_box_tensor = &input_tensors[0];
    const TfLiteTensor* raw_score_tensor = &input_tensors[1];

    // TODO: Add flexible input tensor size handling.
    CHECK_EQ(raw_box_tensor->dims->size, 3);
    CHECK_EQ(raw_box_tensor->dims->data[0], 1);
    CHECK_EQ(raw_box_tensor->dims->data[1], num_boxes_);
    CHECK_EQ(raw_box_tensor->dims->data[2], num_coords_);
    CHECK_EQ(raw_score_tensor->dims->size, 3);
    CHECK_EQ(raw_score_tensor->dims->data[0], 1);
    CHECK_EQ(raw_score_tensor->dims->data[1], num_boxes_);
    CHECK_EQ(raw_score_tensor->dims->data[2], num_classes_);
    const float* raw_boxes = raw_box_tensor->data.f;
    const float* raw_scores = raw_score_tensor->data.f;

    // TODO: Support other options to load anchors.
    if (!anchors_init_) {
      if (input_tensors.size() == kNumInputTensorsWithAnchors) {
        const TfLiteTensor* anchor_tensor = &input_tensors[2];
        CHECK_EQ(anchor_tensor->dims->size, 2);
        CHECK_EQ(anchor_tensor->dims->data[0], num_boxes_);
        CHECK_EQ(anchor_tensor->dims->data[1], kNumCoordsPerBox);
        const float* raw_anchors = anchor_tensor->data.f;
        ConvertRawValuesToAnchors(raw_anchors, num_boxes_, &anchors_);
      } else if (side_packet_anchors_) {
        CHECK(!cc->InputSidePackets().Tag("ANCHORS").IsEmpty());
        anchors_ =
            cc->InputSidePackets().Tag("ANCHORS").Get<std::vector<Anchor>>();
      } else {
        return ::mediapipe::UnavailableError("No anchor data available.");
      }
      anchors_init_ = true;
    }
    std::vector<float> boxes(num_boxes_ * num_coords_);
    RETURN_IF_ERROR(DecodeBoxes(raw_boxes, anchors_, &boxes));

    std::vector<float> detection_scores(num_boxes_);
    std::vector<int> detection_classes(num_boxes_);

    // Filter classes by scores.
    for (int i = 0; i < num_boxes_; ++i) {
      int class_id = -1;
      float max_score = -std::numeric_limits<float>::max();
      // Find the top score for box i.
      for (int score_idx = 0; score_idx < num_classes_; ++score_idx) {
        if (ignore_classes_.find(score_idx) == ignore_classes_.end()) {
          auto score = raw_scores[i * num_classes_ + score_idx];
          if (options_.sigmoid_score()) {
            if (options_.has_score_clipping_thresh()) {
              score = score < -options_.score_clipping_thresh()
                          ? -options_.score_clipping_thresh()
                          : score;
              score = score > options_.score_clipping_thresh()
                          ? options_.score_clipping_thresh()
                          : score;
            }
            score = 1.0f / (1.0f + std::exp(-score));
          }
          if (max_score < score) {
            max_score = score;
            class_id = score_idx;
          }
        }
      }
      detection_scores[i] = max_score;
      detection_classes[i] = class_id;
    }

    RETURN_IF_ERROR(ConvertToDetections(boxes.data(), detection_scores.data(),
                                        detection_classes.data(),
                                        output_detections));
  } else {
    // Postprocessing on CPU with postprocessing op (e.g. anchor decoding and
    // non-maximum suppression) within the model.
    RET_CHECK_EQ(input_tensors.size(), 4);

    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "Passed check 000");  
    /*const TfLiteTensor* detection_boxes_tensor = &input_tensors[0];
    const TfLiteTensor* detection_classes_tensor = &input_tensors[1];
    const TfLiteTensor* detection_scores_tensor = &input_tensors[2];
    const TfLiteTensor* num_boxes_tensor = &input_tensors[3];
    
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_boxes_tensor->dims->size - %d ", detection_boxes_tensor->dims->size);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_boxes_tensor->dims->size - %d ", detection_boxes_tensor->dims->data[0]);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_boxes_tensor->dims->size - %d ", detection_boxes_tensor->dims->data[1]);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_boxes_tensor->dims->size - %d ", detection_boxes_tensor->dims->data[2]);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_boxes_tensor->dims->size - %d ", detection_boxes_tensor->dims->data[3]);
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_boxes_tensor->is_var - %s ", detection_boxes_tensor->is_variable ? "true" : "false");
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_boxes_tensor->type - %d ", detection_boxes_tensor->type); 

    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_classes_tensor->dims->size - %d ", detection_classes_tensor->dims->size);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_classes_tensor->dims->size - %d ", detection_classes_tensor->dims->data[0]);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_classes_tensor->dims->size - %d ", detection_classes_tensor->dims->data[1]);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_classes_tensor->dims->size - %d ", detection_classes_tensor->dims->data[2]);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_classes_tensor->dims->size - %d ", detection_classes_tensor->dims->data[3]); 
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_classes_tensor->type - %d ", detection_classes_tensor->type); 

    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_scores_tensor->dims->size - %d ", detection_scores_tensor->dims->size);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_scores_tensor->dims->size - %d ", detection_scores_tensor->dims->data[0]);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_scores_tensor->dims->size - %d ", detection_scores_tensor->dims->data[1]);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_scores_tensor->dims->size - %d ", detection_scores_tensor->dims->data[2]);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_scores_tensor->dims->size - %d ", detection_scores_tensor->dims->data[3]); 
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_scores_tensor->type - %d ", detection_scores_tensor->type); 

    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "num_boxes_tensor->dims->size - %d ", num_boxes_tensor->dims->size);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "num_boxes_tensor->dims->size - %d ", num_boxes_tensor->dims->data[0]);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "num_boxes_tensor->dims->size - %d ", num_boxes_tensor->dims->data[1]);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "num_boxes_tensor->dims->size - %d ", num_boxes_tensor->dims->data[2]);  
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "num_boxes_tensor->dims->size - %d ", num_boxes_tensor->dims->data[3]); 
    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "num_boxes_tensor->type - %d ", num_boxes_tensor->type);  


    for (int i = 0; i < 9 * 9 * 17; ++i) {
      if (i % 17 == 0) {
        __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "----------------------");  
      }
      __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_boxes_tensor->data - %f ", detection_boxes_tensor->data.f[i]);
    }
    for (int i = 0; i < 9 * 9 * 17; ++i) {
      __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_classes_tensor->data - %f ", detection_classes_tensor->data.f[i]);
    }
    for (int i = 0; i < 9 * 9 * 17; ++i) {
      __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "detection_scores_tensor->data - %f ", detection_scores_tensor->data.f[i]);
    }
    for (int i = 0; i < 9 * 9 * 17; ++i) {
      __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "num_boxes_tensor->data - %f ", num_boxes_tensor->data.f[i]);
    }*/ 

    const TfLiteTensor* part_heatmap_tensor = &input_tensors[0];
    const TfLiteTensor* point_heatmap_tensor = &input_tensors[1];
    const TfLiteTensor* point_offsets_tensor = &input_tensors[2];
    const TfLiteTensor* raw_segment_tensor = &input_tensors[3];
    const float* part_heatmap = reinterpret_cast<const float*>(part_heatmap_tensor->data.raw);
    const float* point_heatmap = reinterpret_cast<const float*>(point_heatmap_tensor->data.raw);
    const float* point_offsets = reinterpret_cast<const float*>(point_offsets_tensor->data.raw);
    const float* raw_segment = reinterpret_cast<const float*>(raw_segment_tensor->data.raw);

    std::unique_ptr<ImageFrame> segmentation_frame = 
        absl::make_unique<ImageFrame>(ImageFormat::GRAY8, WIDTH_FULL, HEIGHT_FULL);
    uchar* segmentation_ptr = segmentation_frame->MutablePixelData();
    DecodeSegmentation(part_heatmap, raw_segment, segmentation_ptr);

    std::unique_ptr<std::vector<Keypoint>> keypoints_vec = 
        absl::make_unique<std::vector<Keypoint>>(NUM_KEYPOINTS);
    DecodePose(point_heatmap, point_offsets, keypoints_vec->data());

    if (cc->Outputs().HasTag("VIZ")) {
      uchar* seg_ptr = segmentation_frame->MutablePixelData();
      std::unique_ptr<ImageFrame> viz_output = 
          absl::make_unique<ImageFrame>(ImageFormat::SRGBA, WIDTH_FULL, HEIGHT_FULL);
      uchar* viz_ptr = viz_output->MutablePixelData();
      const int scale = 255 / NUM_CLASSES;
      for (int row = 0; row < HEIGHT_FULL; ++row) {
        for (int col = 0; col < WIDTH_FULL; ++col) {
          uchar pix = *seg_ptr++;
          for (int c = 0; c < 3; ++c) {
            *viz_ptr++ = pix * scale;
          }
          *viz_ptr++ = 255;
        }
      }
      cc->Outputs().Tag("VIZ").Add(viz_output.release(), cc->InputTimestamp());
    }

    if (cc->Outputs().HasTag("SEGMENTATION")) {
      cc->Outputs().Tag("SEGMENTATION").Add(segmentation_frame.release(), cc->InputTimestamp());
    }
    if (cc->Outputs().HasTag("KEYPOINTS")) {
      cc->Outputs().Tag("KEYPOINTS").Add(keypoints_vec.release(), cc->InputTimestamp());
    }

  }
  return ::mediapipe::OkStatus();
}
::mediapipe::Status PoseNetTensorsToDetectionsCalculator::ProcessGPU(
    CalculatorContext* cc, std::vector<Detection>* output_detections) {
#if defined(__ANDROID__)
  const auto& input_tensors =
      cc->Inputs().Tag("TENSORS_GPU").Get<std::vector<GlBuffer>>();

  // Copy inputs.
  tflite::gpu::gl::CopyBuffer(input_tensors[0], *raw_boxes_buffer_.get());
  tflite::gpu::gl::CopyBuffer(input_tensors[1], *raw_scores_buffer_.get());
  if (!anchors_init_) {
    if (side_packet_anchors_) {
      CHECK(!cc->InputSidePackets().Tag("ANCHORS").IsEmpty());
      const auto& anchors =
          cc->InputSidePackets().Tag("ANCHORS").Get<std::vector<Anchor>>();
      std::vector<float> raw_anchors(num_boxes_ * kNumCoordsPerBox);
      ConvertAnchorsToRawValues(anchors, num_boxes_, raw_anchors.data());
      raw_anchors_buffer_->Write<float>(absl::MakeSpan(raw_anchors));
    } else {
      CHECK_EQ(input_tensors.size(), 3);
      tflite::gpu::gl::CopyBuffer(input_tensors[2], *raw_anchors_buffer_.get());
    }
    anchors_init_ = true;
  }

  // Run shaders.
  RETURN_IF_ERROR(gpu_helper_.RunInGlContext(
      [this, &input_tensors]() -> ::mediapipe::Status {
        // Decode boxes.
        decoded_boxes_buffer_->BindToIndex(0);
        raw_boxes_buffer_->BindToIndex(1);
        raw_anchors_buffer_->BindToIndex(2);
        const tflite::gpu::uint3 decode_workgroups = {num_boxes_, 1, 1};
        decode_program_->Dispatch(decode_workgroups);

        // Score boxes.
        scored_boxes_buffer_->BindToIndex(0);
        raw_scores_buffer_->BindToIndex(1);
        const tflite::gpu::uint3 score_workgroups = {num_boxes_, 1, 1};
        score_program_->Dispatch(score_workgroups);

        return ::mediapipe::OkStatus();
      }));

  // Copy decoded boxes from GPU to CPU.
  std::vector<float> boxes(num_boxes_ * num_coords_);
  auto status = decoded_boxes_buffer_->Read(absl::MakeSpan(boxes));
  if (!status.ok()) {
    return ::mediapipe::InternalError(status.error_message());
  }
  std::vector<float> score_class_id_pairs(num_boxes_ * 2);
  status = scored_boxes_buffer_->Read(absl::MakeSpan(score_class_id_pairs));
  if (!status.ok()) {
    return ::mediapipe::InternalError(status.error_message());
  }

  // TODO: b/138851969. Is it possible to output a float vector
  // for score and an int vector for class so that we can avoid copying twice?
  std::vector<float> detection_scores(num_boxes_);
  std::vector<int> detection_classes(num_boxes_);
  for (int i = 0; i < num_boxes_; ++i) {
    detection_scores[i] = score_class_id_pairs[i * 2];
    detection_classes[i] = static_cast<int>(score_class_id_pairs[i * 2 + 1]);
  }
  RETURN_IF_ERROR(ConvertToDetections(boxes.data(), detection_scores.data(),
                                      detection_classes.data(),
                                      output_detections));
#else
  LOG(ERROR) << "GPU input on non-Android not supported yet.";
#endif  // defined(__ANDROID__)
  return ::mediapipe::OkStatus();
}

::mediapipe::Status PoseNetTensorsToDetectionsCalculator::Close(
    CalculatorContext* cc) {
#if defined(__ANDROID__)
  gpu_helper_.RunInGlContext([this] {
    decode_program_.reset();
    score_program_.reset();
    decoded_boxes_buffer_.reset();
    raw_boxes_buffer_.reset();
    raw_anchors_buffer_.reset();
    scored_boxes_buffer_.reset();
    raw_scores_buffer_.reset();
  });
#endif  // __ANDROID__

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PoseNetTensorsToDetectionsCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ =
      cc->Options<::mediapipe::PoseNetTensorsToDetectionsCalculatorOptions>();

  num_classes_ = options_.num_classes();
  num_boxes_ = options_.num_boxes();
  num_coords_ = options_.num_coords();

  // Currently only support 2D when num_values_per_keypoint equals to 2.
  CHECK_EQ(options_.num_values_per_keypoint(), 2);

  // Check if the output size is equal to the requested boxes and keypoints.
  CHECK_EQ(options_.num_keypoints() * options_.num_values_per_keypoint() +
               kNumCoordsPerBox,
           num_coords_);

  for (int i = 0; i < options_.ignore_classes_size(); ++i) {
    ignore_classes_.insert(options_.ignore_classes(i));
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PoseNetTensorsToDetectionsCalculator::DecodeBoxes(
    const float* raw_boxes, const std::vector<Anchor>& anchors,
    std::vector<float>* boxes) {
  for (int i = 0; i < num_boxes_; ++i) {
    const int box_offset = i * num_coords_ + options_.box_coord_offset();

    float y_center = raw_boxes[box_offset];
    float x_center = raw_boxes[box_offset + 1];
    float h = raw_boxes[box_offset + 2];
    float w = raw_boxes[box_offset + 3];
    if (options_.reverse_output_order()) {
      x_center = raw_boxes[box_offset];
      y_center = raw_boxes[box_offset + 1];
      w = raw_boxes[box_offset + 2];
      h = raw_boxes[box_offset + 3];
    }

    x_center =
        x_center / options_.x_scale() * anchors[i].w() + anchors[i].x_center();
    y_center =
        y_center / options_.y_scale() * anchors[i].h() + anchors[i].y_center();

    if (options_.apply_exponential_on_box_size()) {
      h = std::exp(h / options_.h_scale()) * anchors[i].h();
      w = std::exp(w / options_.w_scale()) * anchors[i].w();
    } else {
      h = h / options_.h_scale() * anchors[i].h();
      w = w / options_.w_scale() * anchors[i].w();
    }

    const float ymin = y_center - h / 2.f;
    const float xmin = x_center - w / 2.f;
    const float ymax = y_center + h / 2.f;
    const float xmax = x_center + w / 2.f;

    (*boxes)[i * num_coords_ + 0] = ymin;
    (*boxes)[i * num_coords_ + 1] = xmin;
    (*boxes)[i * num_coords_ + 2] = ymax;
    (*boxes)[i * num_coords_ + 3] = xmax;

    if (options_.num_keypoints()) {
      for (int k = 0; k < options_.num_keypoints(); ++k) {
        const int offset = i * num_coords_ + options_.keypoint_coord_offset() +
                           k * options_.num_values_per_keypoint();

        float keypoint_y = raw_boxes[offset];
        float keypoint_x = raw_boxes[offset + 1];
        if (options_.reverse_output_order()) {
          keypoint_x = raw_boxes[offset];
          keypoint_y = raw_boxes[offset + 1];
        }

        (*boxes)[offset] = keypoint_x / options_.x_scale() * anchors[i].w() +
                           anchors[i].x_center();
        (*boxes)[offset + 1] =
            keypoint_y / options_.y_scale() * anchors[i].h() +
            anchors[i].y_center();
      }
    }
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status PoseNetTensorsToDetectionsCalculator::ConvertToDetections(
    const float* detection_boxes, const float* detection_scores,
    const int* detection_classes, std::vector<Detection>* output_detections) {
  for (int i = 0; i < num_boxes_; ++i) {
    if (options_.has_min_score_thresh() &&
        detection_scores[i] < options_.min_score_thresh()) {
      continue;
    }
    const int box_offset = i * num_coords_;
    Detection detection = ConvertToDetection(
        detection_boxes[box_offset + 0], detection_boxes[box_offset + 1],
        detection_boxes[box_offset + 2], detection_boxes[box_offset + 3],
        detection_scores[i], detection_classes[i], options_.flip_vertically());
    // Add keypoints.
    if (options_.num_keypoints() > 0) {
      auto* location_data = detection.mutable_location_data();
      for (int kp_id = 0; kp_id < options_.num_keypoints() *
                                      options_.num_values_per_keypoint();
           kp_id += options_.num_values_per_keypoint()) {
        auto keypoint = location_data->add_relative_keypoints();
        const int keypoint_index =
            box_offset + options_.keypoint_coord_offset() + kp_id;
        keypoint->set_x(detection_boxes[keypoint_index + 0]);
        keypoint->set_y(options_.flip_vertically()
                            ? 1.f - detection_boxes[keypoint_index + 1]
                            : detection_boxes[keypoint_index + 1]);
      }
    }
    output_detections->emplace_back(detection);
  }
  return ::mediapipe::OkStatus();
}

Detection PoseNetTensorsToDetectionsCalculator::ConvertToDetection(
    float box_ymin, float box_xmin, float box_ymax, float box_xmax, float score,
    int class_id, bool flip_vertically) {
  Detection detection;
  detection.add_score(score);
  detection.add_label_id(class_id);

  LocationData* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);

  LocationData::RelativeBoundingBox* relative_bbox =
      location_data->mutable_relative_bounding_box();

  relative_bbox->set_xmin(box_xmin);
  relative_bbox->set_ymin(flip_vertically ? 1.f - box_ymax : box_ymin);
  relative_bbox->set_width(box_xmax - box_xmin);
  relative_bbox->set_height(box_ymax - box_ymin);
  return detection;
}

::mediapipe::Status PoseNetTensorsToDetectionsCalculator::GlSetup(
    CalculatorContext* cc) {
#if defined(__ANDROID__)
  // A shader to decode detection boxes.
  const std::string decode_src = absl::Substitute(
      R"( #version 310 es

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(location = 0) uniform vec4 scale;

layout(std430, binding = 0) writeonly buffer Output {
  float data[];
} boxes;

layout(std430, binding = 1) readonly buffer Input0 {
  float data[];
} raw_boxes;

layout(std430, binding = 2) readonly buffer Input1 {
  float data[];
} raw_anchors;

uint num_coords = uint($0);
int reverse_output_order = int($1);
int apply_exponential = int($2);
int box_coord_offset = int($3);
int num_keypoints = int($4);
int keypt_coord_offset = int($5);
int num_values_per_keypt = int($6);

void main() {
  uint g_idx = gl_GlobalInvocationID.x;  // box index
  uint box_offset = g_idx * num_coords + uint(box_coord_offset);
  uint anchor_offset = g_idx * uint(4);  // check kNumCoordsPerBox

  float y_center, x_center, h, w;

  if (reverse_output_order == int(0)) {
    y_center = raw_boxes.data[box_offset + uint(0)];
    x_center = raw_boxes.data[box_offset + uint(1)];
    h = raw_boxes.data[box_offset + uint(2)];
    w = raw_boxes.data[box_offset + uint(3)];
  } else {
    x_center = raw_boxes.data[box_offset + uint(0)];
    y_center = raw_boxes.data[box_offset + uint(1)];
    w = raw_boxes.data[box_offset + uint(2)];
    h = raw_boxes.data[box_offset + uint(3)];
  }

  float anchor_yc = raw_anchors.data[anchor_offset + uint(0)];
  float anchor_xc = raw_anchors.data[anchor_offset + uint(1)];
  float anchor_h  = raw_anchors.data[anchor_offset + uint(2)];
  float anchor_w  = raw_anchors.data[anchor_offset + uint(3)];

  x_center = x_center / scale.x * anchor_w + anchor_xc;
  y_center = y_center / scale.y * anchor_h + anchor_yc;

  if (apply_exponential == int(1)) {
    h = exp(h / scale.w) * anchor_h;
    w = exp(w / scale.z) * anchor_w;
  } else {
    h = (h / scale.w) * anchor_h;
    w = (w / scale.z) * anchor_w;
  }

  float ymin = y_center - h / 2.0;
  float xmin = x_center - w / 2.0;
  float ymax = y_center + h / 2.0;
  float xmax = x_center + w / 2.0;

  boxes.data[box_offset + uint(0)] = ymin;
  boxes.data[box_offset + uint(1)] = xmin;
  boxes.data[box_offset + uint(2)] = ymax;
  boxes.data[box_offset + uint(3)] = xmax;

  if (num_keypoints > int(0)){
    for (int k = 0; k < num_keypoints; ++k) {
      int kp_offset =
          int(g_idx * num_coords) + keypt_coord_offset + k * num_values_per_keypt;
      float kp_y, kp_x;
      if (reverse_output_order == int(0)) {
        kp_y = raw_boxes.data[kp_offset + int(0)];
        kp_x = raw_boxes.data[kp_offset + int(1)];
      } else {
        kp_x = raw_boxes.data[kp_offset + int(0)];
        kp_y = raw_boxes.data[kp_offset + int(1)];
      }
      boxes.data[kp_offset + int(0)] = kp_x / scale.x * anchor_w + anchor_xc;
      boxes.data[kp_offset + int(1)] = kp_y / scale.y * anchor_h + anchor_yc;
    }
  }
})",
      options_.num_coords(),  // box xywh
      options_.reverse_output_order() ? 1 : 0,
      options_.apply_exponential_on_box_size() ? 1 : 0,
      options_.box_coord_offset(), options_.num_keypoints(),
      options_.keypoint_coord_offset(), options_.num_values_per_keypoint());

  // Shader program
  GlShader decode_shader;
  auto status =
      GlShader::CompileShader(GL_COMPUTE_SHADER, decode_src, &decode_shader);
  if (!status.ok()) {
    return ::mediapipe::InternalError(status.error_message());
  }
  decode_program_ = absl::make_unique<GlProgram>();
  status = GlProgram::CreateWithShader(decode_shader, decode_program_.get());
  if (!status.ok()) {
    return ::mediapipe::InternalError(status.error_message());
  }
  // Outputs
  size_t decoded_boxes_length = num_boxes_ * num_coords_;
  decoded_boxes_buffer_ = absl::make_unique<GlBuffer>();
  status = CreateReadWriteShaderStorageBuffer<float>(
      decoded_boxes_length, decoded_boxes_buffer_.get());
  if (!status.ok()) {
    return ::mediapipe::InternalError(status.error_message());
  }
  // Inputs
  size_t raw_boxes_length = num_boxes_ * num_coords_;
  raw_boxes_buffer_ = absl::make_unique<GlBuffer>();
  status = CreateReadWriteShaderStorageBuffer<float>(raw_boxes_length,
                                                     raw_boxes_buffer_.get());
  if (!status.ok()) {
    return ::mediapipe::InternalError(status.error_message());
  }
  size_t raw_anchors_length = num_boxes_ * kNumCoordsPerBox;
  raw_anchors_buffer_ = absl::make_unique<GlBuffer>();
  status = CreateReadWriteShaderStorageBuffer<float>(raw_anchors_length,
                                                     raw_anchors_buffer_.get());
  if (!status.ok()) {
    return ::mediapipe::InternalError(status.error_message());
  }
  // Parameters
  glUseProgram(decode_program_->id());
  glUniform4f(0, options_.x_scale(), options_.y_scale(), options_.w_scale(),
              options_.h_scale());

  // A shader to score detection boxes.
  const std::string score_src = absl::Substitute(
      R"( #version 310 es

layout(local_size_x = 1, local_size_y = $0, local_size_z = 1) in;

#define FLT_MAX 1.0e+37

shared float local_scores[$0];

layout(std430, binding = 0) writeonly buffer Output {
  float data[];
} scored_boxes;

layout(std430, binding = 1) readonly buffer Input0 {
  float data[];
} raw_scores;

uint num_classes = uint($0);
int apply_sigmoid = int($1);
int apply_clipping_thresh = int($2);
float clipping_thresh = float($3);
int ignore_class_0 = int($4);

float optional_sigmoid(float x) {
  if (apply_sigmoid == int(0)) return x;
  if (apply_clipping_thresh == int(1)) {
    x = clamp(x, -clipping_thresh, clipping_thresh);
  }
  x = 1.0 / (1.0 + exp(-x));
  return x;
}

void main() {
  uint g_idx = gl_GlobalInvocationID.x;   // box idx
  uint s_idx =  gl_LocalInvocationID.y;   // score/class idx

  // load all scores into shared memory
  float score = raw_scores.data[g_idx * num_classes + s_idx];
  local_scores[s_idx] = optional_sigmoid(score);
  memoryBarrierShared();
  barrier();

  // find max score in shared memory
  if (s_idx == uint(0)) {
    float max_score = -FLT_MAX;
    float max_class = -1.0;
    for (int i=ignore_class_0; i<int(num_classes); ++i) {
      if (local_scores[i] > max_score) {
        max_score = local_scores[i];
        max_class = float(i);
      }
    }
    scored_boxes.data[g_idx * uint(2) + uint(0)] = max_score;
    scored_boxes.data[g_idx * uint(2) + uint(1)] = max_class;
  }
})",
      num_classes_, options_.sigmoid_score() ? 1 : 0,
      options_.has_score_clipping_thresh() ? 1 : 0,
      options_.has_score_clipping_thresh() ? options_.score_clipping_thresh()
                                           : 0,
      ignore_classes_.size() ? 1 : 0);

  // # filter classes supported is hardware dependent.
  int max_wg_size;  //  typically <= 1024
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &max_wg_size);  // y-dim
  CHECK_LT(num_classes_, max_wg_size) << "# classes must be < " << max_wg_size;
  // TODO support better filtering.
  CHECK_LE(ignore_classes_.size(), 1) << "Only ignore class 0 is allowed";

  // Shader program
  GlShader score_shader;
  status = GlShader::CompileShader(GL_COMPUTE_SHADER, score_src, &score_shader);
  if (!status.ok()) {
    return ::mediapipe::InternalError(status.error_message());
  }
  score_program_ = absl::make_unique<GlProgram>();
  status = GlProgram::CreateWithShader(score_shader, score_program_.get());
  if (!status.ok()) {
    return ::mediapipe::InternalError(status.error_message());
  }
  // Outputs
  size_t scored_boxes_length = num_boxes_ * 2;  // score, class
  scored_boxes_buffer_ = absl::make_unique<GlBuffer>();
  status = CreateReadWriteShaderStorageBuffer<float>(
      scored_boxes_length, scored_boxes_buffer_.get());
  if (!status.ok()) {
    return ::mediapipe::InternalError(status.error_message());
  }
  // Inputs
  size_t raw_scores_length = num_boxes_ * num_classes_;
  raw_scores_buffer_ = absl::make_unique<GlBuffer>();
  status = CreateReadWriteShaderStorageBuffer<float>(raw_scores_length,
                                                     raw_scores_buffer_.get());
  if (!status.ok()) {
    return ::mediapipe::InternalError(status.error_message());
  }

#endif  // defined(__ANDROID__)
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe

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

#include <cmath>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/port/ret_check.h"
#include "tensorflow/lite/interpreter.h"

#if defined(__ANDROID__)
#include <android/log.h>
#endif  // ANDROID

namespace mediapipe {

class PoseDetectionLetterboxRemovalCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("TENSORS").Set<std::vector<TfLiteTensor>>();
    cc->Inputs().Tag("LETTERBOX_PADDING").Set<std::array<float, 4>>();

    cc->Outputs().Tag("DETECTIONS").Set<std::vector<Detection>>();

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {

    __android_log_print(
      ANDROID_LOG_INFO, "debug_yichuc", "Pose DetectionLetterboxRemovalCalculator input-time %ld",
      cc->InputTimestamp().Microseconds());

    // Process detected tensors from Posenet.
    const auto& input_pose_tensors =
        cc->Inputs().Tag("TENSORS").Get<std::vector<TfLiteTensor>>();
    const auto& letterbox_padding =
        cc->Inputs().Tag("LETTERBOX_PADDING").Get<std::array<float, 4>>();
    const TfLiteTensor* heatmap_tensor = &input_pose_tensors[0];
    const TfLiteTensor* offset_tensor = &input_pose_tensors[1];
    const float* heatmap = reinterpret_cast<const float*>(heatmap_tensor->data.raw);
    const float* offsetmap = reinterpret_cast<const float*>(offset_tensor->data.raw);

    const int kOutputStride = 32;
    const int kImageWidth = 257;
    const int kImageHeight = 257;
    const int kOrigImageWidth = 2560;
    const int kOrigImageHeight = 1440;
    const int kHeatMapWidth = 9;
    const int kHeatMapHeight = 9;
    const int kOffsetMapWidth = 9;
    const int kOffsetMapHeight = 9;
    const int kNumLayers = 17;
    const int kOffsetLayers = kNumLayers * 2;

    std::vector<std::pair<int, int>> max_heatmap_positions(kNumLayers);
    std::vector<std::pair<int, int>> key_points_of_all_parts(kNumLayers);

    // Allocate output structure.
    auto output_detection_pose = absl::make_unique<std::vector<Detection>>();
    // Process letter padding box.
    const float pad_left = letterbox_padding[0];
    const float pad_top = letterbox_padding[1];
    const float pad_left_and_right = letterbox_padding[0] + letterbox_padding[2];
    const float pad_top_and_bottom = letterbox_padding[1] + letterbox_padding[3];

    for (int k = 0; k < kNumLayers; ++k) {
      float max_value = std::numeric_limits<float>::lowest();
      std::pair<int, int> max_pos = {-1, -1};
      for (int i = 0; i < kHeatMapHeight; ++i) {
        for (int j = 0; j < kHeatMapWidth; ++j) {
          int index = (i * kHeatMapWidth + j) * kNumLayers + k;
          //__android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "heat_map index %d, value %.4f", index, heatmap[index]);
          if (heatmap[index] > max_value) {
            max_value = heatmap[index];
            max_pos = std::make_pair(i, j);
          }
        }
      }
      max_heatmap_positions[k] = max_pos;

      int heat_y = max_pos.first;
      int heat_x = max_pos.second;
      int offset_y_index = (heat_y * kOffsetMapWidth + heat_x) * kOffsetLayers + k;
      int offset_x_index = (heat_y * kOffsetMapWidth + heat_x) * kOffsetLayers + k + 17;

      int keypoint_y = static_cast<int>(heat_y * kOutputStride + offsetmap[offset_y_index]);
      int keypoint_x = static_cast<int>(heat_x * kOutputStride + offsetmap[offset_x_index]);
      key_points_of_all_parts[k] = {keypoint_y, keypoint_x};
      //__android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "absolute_key_points_of_all_parts k = %d, y=%d, x=%d ",
      //  k, keypoint_y, keypoint_x);

      // Assign into output structure.
      float relative_keypoint_x = 1.0f * keypoint_x / kImageWidth;
      float relative_keypoint_y = 1.0f * keypoint_y / kImageHeight;

      Detection detection_pose;
      detection_pose.mutable_location_data()->mutable_relative_bounding_box()->set_xmin(
          (relative_keypoint_x - pad_left) / (1.0f - pad_left_and_right) );
      detection_pose.mutable_location_data()->mutable_relative_bounding_box()->set_ymin(
          (relative_keypoint_y - pad_top) / (1.0f - pad_top_and_bottom));
      detection_pose.mutable_location_data()->mutable_relative_bounding_box()->set_width(0);
      detection_pose.mutable_location_data()->mutable_relative_bounding_box()->set_height(0);
      output_detection_pose->push_back(detection_pose);
    }

    cc->Outputs()
        .Tag("DETECTIONS")
        .Add(output_detection_pose.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(PoseDetectionLetterboxRemovalCalculator);

}  // namespace mediapipe

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

namespace {

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
const int kNumEdges = kNumLayers - 1;
const int kDispLayers = kNumEdges * 2;

struct KeyPoint {
  int x;
  int y;
  float score;
};

struct Pose {
  std::map<int, KeyPoint> key_points;
  float instance_score;
};

bool WithinNmsRadiusOfCorrespondingPoint(
  const std::vector<Pose>& poses, int sq_nms_radius, int x, int y, int keypoint_id) {
  for (const auto& pose : poses) {
    if (!pose.key_points.count(keypoint_id)) {
      continue;
    }
    if ((pose.key_points.at(keypoint_id).x - x) * (pose.key_points.at(keypoint_id).x - x) +
        (pose.key_points.at(keypoint_id).y - y) * (pose.key_points.at(keypoint_id).y - y)
        <= sq_nms_radius) {
      return true;
    }
  }
  return false;
}

float GetInstanceScore(
  const std::vector<Pose>& existing_poses, int sq_nms_radius,
  const Pose& query_instance) {
  int num_count = 0;
  float result = 0.0;
  for (auto key_point_itr = query_instance.key_points.begin();
       key_point_itr != query_instance.key_points.end(); ++key_point_itr) {
    if (!WithinNmsRadiusOfCorrespondingPoint(
         existing_poses, sq_nms_radius, key_point_itr->second.x, key_point_itr->second.y,
         key_point_itr->first)) {
      result += key_point_itr->second.score;
      ++num_count;
    }
  }

  if (num_count == 0) {
    return 0;
  } else {
    return result / num_count;
  }
}

struct ScoreElement {
  float score;
  int heat_map_y;
  int heat_map_x;
  int keypoint_id;
};

bool ScoreIsMaximumInLocalWindow(
  int keypoint_id, float score, int heat_map_y, int heat_map_x, int local_maximum_radius,
  const float* heatmap) {
  bool is_local_maximum = true;
  int y_start = std::max(0, heat_map_y - local_maximum_radius);
  int y_end = std::min(kHeatMapHeight, heat_map_y + local_maximum_radius + 1);
  for (int y_curr = y_start; y_curr < y_end; ++y_curr) {
    int x_start = std::max(0, heat_map_x - local_maximum_radius);
    int x_end = std::min(kHeatMapWidth, heat_map_x + local_maximum_radius);
    for (int x_curr = x_start; x_curr < x_end; ++x_curr) {
      int index = (y_curr * kHeatMapWidth + x_curr) * kNumLayers + keypoint_id;
      if (heatmap[index] > score) {
        is_local_maximum = false;
        break;
      }
    }
    if (!is_local_maximum) {
      break;
    }
  }
  return is_local_maximum;
}

KeyPoint GetImageCoords(const ScoreElement& score_element, const float* offsetmap) {
  int offset_y_index = (score_element.heat_map_y * kOffsetMapWidth + score_element.heat_map_x) * kOffsetLayers + score_element.keypoint_id;
  int offset_x_index = (score_element.heat_map_y * kOffsetMapWidth + score_element.heat_map_x) * kOffsetLayers + score_element.keypoint_id + 17;

  KeyPoint key_pt;
  key_pt.y = static_cast<int>(score_element.heat_map_y * kOutputStride + offsetmap[offset_y_index]);
  key_pt.x = static_cast<int>(score_element.heat_map_x * kOutputStride + offsetmap[offset_x_index]);

  return key_pt;
}

std::deque<ScoreElement> BuildPartWithHeatQueue(
  const float* heatmap, float heat_threshold, int local_maximum_radius) {
  std::deque<ScoreElement> result_queue;
  for (int i = 0; i < kHeatMapHeight; ++i) {
    for (int j = 0; j < kHeatMapWidth; ++j) {
      for (int keypoint_id = 0; keypoint_id < kNumLayers; ++keypoint_id) {
        int index = (i * kHeatMapWidth + j) * kNumLayers + keypoint_id;
        float score = heatmap[index];
        if (score < heat_threshold) {
          continue;
        }
        if (ScoreIsMaximumInLocalWindow(
          keypoint_id, score, i, j, local_maximum_radius, heatmap)) {
          ScoreElement element;
          element.score = score;
          element.heat_map_y = i;
          element.heat_map_x = j;
          element.keypoint_id = keypoint_id;
          result_queue.push_back(element);
        }
      }
    }
  }

  return result_queue;
}

std::pair<float, float> GetDisplacement(
  int edge_id, const std::pair<int, int>& keypt_coord, const float* disp_map) {
  int y = keypt_coord.first;
  int x = keypt_coord.second;

  int index_y = (y * kHeatMapWidth + x) * kDispLayers + edge_id;
  int index_x = (y * kHeatMapWidth + x) * kDispLayers + edge_id + kNumEdges;

  return std::make_pair(disp_map[index_y], disp_map[index_x]);
}

std::pair<int, int> GetStridedIndexNearPoint(
  const KeyPoint& keypt_coord, int height, int width) {
  int val_y = static_cast<int>(std::round(1. * keypt_coord.y / kOutputStride));
  if (val_y < 0) val_y = 0;
  if (val_y >= height) val_y = height - 1;

  int val_x = static_cast<int>(std::round(1. * keypt_coord.x / kOutputStride));
  if (val_x < 0) val_x = 0;
  if (val_x >= width) val_x = width - 1;

  return std::make_pair(val_y, val_x);
}

std::pair<std::map<int, std::vector<int>>, std::map<int, int>> BuildPoseChain01() {
  std::map<int, std::vector<int>> parent_to_child;
  std::map<int, int> child_to_parent;

  for (int i = 0; i < kNumLayers; ++i) {
    parent_to_child[i] = std::vector<int>{};
  }
  parent_to_child[0].push_back(1);
  parent_to_child[1].push_back(3);
  parent_to_child[0].push_back(2);
  parent_to_child[2].push_back(4);
  parent_to_child[0].push_back(5);
  parent_to_child[5].push_back(7);
  parent_to_child[7].push_back(9);
  parent_to_child[5].push_back(11);
  parent_to_child[11].push_back(13);
  parent_to_child[13].push_back(15);
  parent_to_child[0].push_back(6);
  parent_to_child[6].push_back(8);
  parent_to_child[8].push_back(10);
  parent_to_child[6].push_back(12);
  parent_to_child[12].push_back(14);
  parent_to_child[14].push_back(16);

  child_to_parent[1] = 0;
  child_to_parent[3] = 1;
  child_to_parent[2] = 0;
  child_to_parent[4] = 2;
  child_to_parent[5] = 0;
  child_to_parent[7] = 5;
  child_to_parent[9] = 7;
  child_to_parent[11] = 5;
  child_to_parent[13] = 11;
  child_to_parent[15] = 13;
  child_to_parent[6] = 0;
  child_to_parent[8] = 6;
  child_to_parent[10] = 8;
  child_to_parent[12] = 6;
  child_to_parent[14] = 12;
  child_to_parent[16] = 14;

  return std::make_pair(parent_to_child, child_to_parent);
}

std::pair<std::vector<int>, std::vector<int>> BuildPoseChain02() {
  std::vector<int> parent_to_child(kNumEdges);
  std::vector<int> child_to_parent(kNumEdges);

  parent_to_child[0] = 0;
  child_to_parent[0] = 1;
  parent_to_child[1] = 1;
  child_to_parent[1] = 3;
  parent_to_child[2] = 0;
  child_to_parent[2] = 2;
  parent_to_child[3] = 2;
  child_to_parent[3] = 4;

  parent_to_child[4] = 0;
  child_to_parent[4] = 5;
  parent_to_child[5] = 5;
  child_to_parent[5] = 7;
  parent_to_child[6] = 7;
  child_to_parent[6] = 9;
  parent_to_child[7] = 5;
  child_to_parent[7] = 11;

  parent_to_child[8] = 11;
  child_to_parent[8] = 13;
  parent_to_child[9] = 13;
  child_to_parent[9] = 15;
  parent_to_child[10] = 0;
  child_to_parent[10] = 6;
  parent_to_child[11] = 6;
  child_to_parent[11] = 8;

  parent_to_child[12] = 8;
  child_to_parent[12] = 10;
  parent_to_child[13] = 6;
  child_to_parent[13] = 12;
  parent_to_child[14] = 12;
  child_to_parent[14] = 14;
  parent_to_child[15] = 14;
  child_to_parent[15] = 16;

  return std::make_pair(parent_to_child, child_to_parent);
}

KeyPoint TraverseToTarget(
  int edge_id, const KeyPoint& source_keypt, int target_keypoint_id, const float* heatmap,
  const float* offsetmap, const float* disp_map) {

  std::pair<int, int> source_keypt_indices = GetStridedIndexNearPoint(source_keypt, kHeatMapHeight, kHeatMapWidth);
  std::pair<float, float> displacement = GetDisplacement(edge_id, source_keypt_indices, disp_map);

  KeyPoint target_keypt = source_keypt;
  target_keypt.y += displacement.first;
  target_keypt.x += displacement.second;

  const int kOffsetRefineStep = 2;
  for (int i = 0; i < kOffsetRefineStep; ++i) {
    std::pair<int, int> target_keypt_indices = GetStridedIndexNearPoint(target_keypt, kHeatMapHeight, kHeatMapWidth);

    int offset_y_index = (target_keypt_indices.first * kOffsetMapWidth + target_keypt_indices.second) * kOffsetLayers + target_keypoint_id;
    int offset_x_index = (target_keypt_indices.first * kOffsetMapWidth + target_keypt_indices.second) * kOffsetLayers + target_keypoint_id + 17;

    target_keypt.y = target_keypt_indices.first * kOutputStride + offsetmap[offset_y_index];
    target_keypt.x = target_keypt_indices.second * kOutputStride + offsetmap[offset_x_index];
  }

  return target_keypt;
}

Pose DecodePose(
  const ScoreElement& score_element, const float* heatmap, const float* offsetmap,
  const float* disp_fwd_map, const float* disp_bwd_map) {
  std::map<int, KeyPoint> instance_keypoints;

  int root_part_id = score_element.keypoint_id;
  float root_score = score_element.score;
  KeyPoint root_coord = GetImageCoords(score_element, offsetmap);
  instance_keypoints[root_part_id] = root_coord;

  auto pose_chain = BuildPoseChain02();
  const std::vector<int>& parent_to_child = pose_chain.first;
  const std::vector<int>& child_to_parent = pose_chain.second;

  for (int edge_id = kNumEdges - 1; edge_id >= 0; --edge_id) {
    const int source_keypoint_id = child_to_parent[edge_id];
    const int target_keypoint_id = parent_to_child[edge_id];
    if (instance_keypoints.count(source_keypoint_id) &&
        !instance_keypoints.count(target_keypoint_id)) {
      instance_keypoints[target_keypoint_id] = TraverseToTarget(
        edge_id, instance_keypoints[source_keypoint_id], target_keypoint_id, heatmap, offsetmap,
        disp_bwd_map);
    }
  }

  for (int edge_id = 0; edge_id < kNumEdges; ++edge_id) {
    const int source_keypoint_id = parent_to_child[edge_id];
    const int target_keypoint_id = child_to_parent[edge_id];
    if (instance_keypoints.count(source_keypoint_id) &&
        !instance_keypoints.count(target_keypoint_id)) {
      instance_keypoints[target_keypoint_id] = TraverseToTarget(
        edge_id, instance_keypoints[source_keypoint_id], target_keypoint_id, heatmap, offsetmap,
        disp_fwd_map);
    }
  }

  Pose ret_pose;
  ret_pose.key_points = instance_keypoints;
  return ret_pose;
}



}  // namespace

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

  ::mediapipe::Status ProcessSinglePose(CalculatorContext* cc) {
    // Process detected tensors from Posenet.
    const auto& input_pose_tensors =
      cc->Inputs().Tag("TENSORS").Get<std::vector<TfLiteTensor>>();
    const auto& letterbox_padding =
      cc->Inputs().Tag("LETTERBOX_PADDING").Get<std::array<float, 4>>();
    const TfLiteTensor* heatmap_tensor = &input_pose_tensors[0];
    const TfLiteTensor* offset_tensor = &input_pose_tensors[1];
    const float* heatmap = reinterpret_cast<const float*>(heatmap_tensor->data.raw);
    const float* offsetmap = reinterpret_cast<const float*>(offset_tensor->data.raw);

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

  ::mediapipe::Status ProcessMutliplePoses(CalculatorContext* cc) {

    // Process detected tensors from Posenet.
    const auto& input_pose_tensors =
      cc->Inputs().Tag("TENSORS").Get<std::vector<TfLiteTensor>>();
    const auto& letterbox_padding =
      cc->Inputs().Tag("LETTERBOX_PADDING").Get<std::array<float, 4>>();

    const TfLiteTensor* heatmap_tensor = &input_pose_tensors[0];
    const TfLiteTensor* offset_tensor = &input_pose_tensors[1];
    const TfLiteTensor* disp_fwd_tensor = &input_pose_tensors[2];
    const TfLiteTensor* disp_bwd_tensor = &input_pose_tensors[3];

    const float* heatmap = reinterpret_cast<const float*>(heatmap_tensor->data.raw);
    const float* offsetmap = reinterpret_cast<const float*>(offset_tensor->data.raw);
    const float* disp_fwd_map = reinterpret_cast<const float*>(disp_fwd_tensor->data.raw);
    const float* disp_bwd_map = reinterpret_cast<const float*>(disp_bwd_tensor->data.raw);

    // parameters
    const int kLocalMaximumRadius = 1;
    const float kHeatScoreThreshold = 0.5;
    const int kSqNmsRadius = 20 * 20;
    const int kMaxPoseDetections = 3;

    std::deque<ScoreElement> heatscore_queue =
      BuildPartWithHeatQueue(heatmap, kHeatScoreThreshold, kLocalMaximumRadius);

    std::vector<Pose> poses;
    while (poses.size() < kMaxPoseDetections && !heatscore_queue.empty()) {
      ScoreElement score_element = heatscore_queue.front(); heatscore_queue.pop_front();
      KeyPoint keypt_coord = GetImageCoords(score_element, offsetmap);
      if (WithinNmsRadiusOfCorrespondingPoint(
          poses, kSqNmsRadius, keypt_coord.x, keypt_coord.y, score_element.keypoint_id)) {
        continue;
      }

      Pose new_pose = DecodePose(score_element, heatmap, offsetmap, disp_fwd_map, disp_bwd_map);
      new_pose.instance_score = GetInstanceScore(poses, kSqNmsRadius, new_pose);
      poses.push_back(new_pose);
    }

    __android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "ProcessMutliplePoses pose size %lu",
                        poses.size());


    // Allocate output structure.
    auto output_detection_pose = absl::make_unique<std::vector<Detection>>();
    // Process letter padding box.
    const float pad_left = letterbox_padding[0];
    const float pad_top = letterbox_padding[1];
    const float pad_left_and_right = letterbox_padding[0] + letterbox_padding[2];
    const float pad_top_and_bottom = letterbox_padding[1] + letterbox_padding[3];

    for (int p = 0; p < poses.size(); ++p) {
      for (int k = 0; k < kNumLayers; ++k) {
        // Assign into output structure.
        float relative_keypoint_x = 1.0f * poses[p].key_points[k].x / kImageWidth;
        float relative_keypoint_y = 1.0f * poses[p].key_points[k].y / kImageHeight;
        Detection detection_pose;
        detection_pose.mutable_location_data()->mutable_relative_bounding_box()->set_xmin(
          (relative_keypoint_x - pad_left) / (1.0f - pad_left_and_right));
        detection_pose.mutable_location_data()->mutable_relative_bounding_box()->set_ymin(
          (relative_keypoint_y - pad_top) / (1.0f - pad_top_and_bottom));
        detection_pose.mutable_location_data()->mutable_relative_bounding_box()->set_width(0);
        detection_pose.mutable_location_data()->mutable_relative_bounding_box()->set_height(0);
        output_detection_pose->push_back(detection_pose);
      }
    }

    cc->Outputs()
      .Tag("DETECTIONS")
      .Add(output_detection_pose.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    // Experiment
    return ProcessMutliplePoses(cc);
    //return ProcessSinglePose(cc);
  }
};
REGISTER_CALCULATOR(PoseDetectionLetterboxRemovalCalculator);

}  // namespace mediapipe

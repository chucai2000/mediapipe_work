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
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "tensorflow/lite/interpreter.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/util/annotation_renderer.h"
#include "mediapipe/framework/formats/detection.pb.h"

#if defined(__ANDROID__)
#include <android/log.h>
#endif  // ANDROID

namespace mediapipe {

class SyncDetectionsCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  ::mediapipe::Status ProcessCPU(CalculatorContext* cc,
                                 std::vector<Detection>* output_detections);

  Timestamp prev_output_timestamp_;

};
REGISTER_CALCULATOR(SyncDetectionsCalculator);

::mediapipe::Status SyncDetectionsCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
    cc->Inputs().Index(i).Set<std::vector<Detection>>();
  }

  if (cc->Outputs().HasTag("RENDER_DATA")) {
    cc->Outputs().Tag("RENDER_DATA").Set<RenderData>();
  }

  if (cc->Inputs().HasTag("VIZ")) {
    cc->Inputs().Tag("VIZ").Set<ImageFrame>();
  }
  if (cc->Outputs().HasTag("VIZ")) {
    cc->Outputs().Tag("VIZ").Set<ImageFrame>();
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status SyncDetectionsCalculator::Open(
    CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  return ::mediapipe::OkStatus();
}

::mediapipe::Status SyncDetectionsCalculator::Process(
    CalculatorContext* cc) {
  auto output_detections = absl::make_unique<std::vector<Detection>>();

  RETURN_IF_ERROR(ProcessCPU(cc, output_detections.get()));

  return ::mediapipe::OkStatus();
}

::mediapipe::Status SyncDetectionsCalculator::ProcessCPU(
    CalculatorContext* cc, std::vector<Detection>* output_detections) {

  __android_log_print(
    ANDROID_LOG_INFO, "debug_yichuc", "SyncDetectionsCalculator input-time %ld, num_entries=%d",
    cc->InputTimestamp().Microseconds(), cc->Inputs().NumEntries());
  if (cc->InputTimestamp() <= prev_output_timestamp_) {
    __android_log_print(
      ANDROID_LOG_INFO, "debug_yichuc", "SyncDetectionsCalculator skipped.");
    return ::mediapipe::OkStatus();
  }

  std::vector<std::vector<std::pair<float, float>>> key_points_of_all_parts;
  if (cc->Inputs().NumEntries() >= 2 && !cc->Inputs().Index(1).IsEmpty()) {
    __android_log_print(
      ANDROID_LOG_INFO, "debug_yichuc", "SyncDetectionsCalculator exist pose");
    const int kNumLayers = 17;
    const int kImageWidth = 257;
    const int kImageHeight = 257;
    const int kOrigImageWidth = 2560;
    const int kOrigImageHeight = 1440;
    const auto& input_pose_detections =
        cc->Inputs().Index(1).Get<std::vector<Detection>>();

    int num_persons = input_pose_detections.size() / kNumLayers;
    __android_log_print(
      ANDROID_LOG_INFO, "debug_yichuc", "SyncDetectionsCalculator num_persons = %d, input_pose_detections.size() = %lu", num_persons, input_pose_detections.size());
    key_points_of_all_parts.resize(num_persons);
    for (int p = 0; p < num_persons; ++p) {
      key_points_of_all_parts[p].resize(kNumLayers);
      for (int k = 0; k < kNumLayers; ++k) {
        key_points_of_all_parts[p][k] = {
            input_pose_detections[p * kNumLayers + k].location_data().relative_bounding_box().ymin(),
            input_pose_detections[p * kNumLayers + k].location_data().relative_bounding_box().xmin()};
        //__android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "relative_key_points_of_all_parts k = %d, y=%.3f, x=%.3f ",
        //    k, key_points_of_all_parts[k].first, key_points_of_all_parts[k].second);
      }
    }
  }

  std::vector<std::vector<double>> face_bounding_boxes;
  if (!cc->Inputs().Index(0).IsEmpty()) {
    __android_log_print(
      ANDROID_LOG_INFO, "debug_yichuc", "SyncDetectionsCalculator exist face");
    const auto& input_face_detections =
      cc->Inputs().Index(0).Get<std::vector<Detection>>();
    for (const auto& face_detection : input_face_detections) {
      const auto& val = face_detection.location_data().relative_bounding_box();
      std::vector<double> face_bounding_box = {val.xmin(), val.ymin(), val.width(), val.height()};
      //__android_log_print(ANDROID_LOG_INFO, "debug_yichuc", "relative_key_points_of_all_parts face, yc=%.3f, xc=%.3f",
      //                    val.ymin() + 0.5f * val.height(), val.xmin() + 0.5f * val.width());
      face_bounding_boxes.emplace_back(face_bounding_box);
    }
  }

  if (cc->Outputs().HasTag("RENDER_DATA")) {
    auto render_data = absl::make_unique<RenderData>();
    float radius = 0.01;

    for (int p = 0; p < key_points_of_all_parts.size(); ++p) {
      for (int k = 0; k < key_points_of_all_parts[p].size(); ++k) {
        auto *pose_annotation = render_data->add_render_annotations();
        auto *location_data_rect = pose_annotation->mutable_rectangle();
        location_data_rect->set_normalized(true);
        location_data_rect->set_left(std::max(0.f, key_points_of_all_parts.at(p).at(k).second - radius));
        location_data_rect->set_top(std::max(0.f, key_points_of_all_parts.at(p).at(k).first - radius));
        location_data_rect->set_right(std::min(1.f, key_points_of_all_parts.at(p).at(k).second + radius));
        location_data_rect->set_bottom(std::min(1.f, key_points_of_all_parts.at(p).at(k).first + radius));

        if (k == 0) {
          pose_annotation->mutable_color()->set_r(255);
          pose_annotation->mutable_color()->set_g(0);
          pose_annotation->mutable_color()->set_b(0);
        } else if (k == 1) {
          pose_annotation->mutable_color()->set_r(0);
          pose_annotation->mutable_color()->set_g(255);
          pose_annotation->mutable_color()->set_b(0);
        } else if (k == 9) {
          pose_annotation->mutable_color()->set_r(0);
          pose_annotation->mutable_color()->set_g(20);
          pose_annotation->mutable_color()->set_b(255);
        } else {
          pose_annotation->mutable_color()->set_r(255);
          pose_annotation->mutable_color()->set_g(255);
          pose_annotation->mutable_color()->set_b(102);
        }
        pose_annotation->set_thickness(3);
      }

      if (key_points_of_all_parts.size() > 0) {
        std::vector<std::pair<int, int>> connect_line = {
          {9,  7},
          {7,  5},
          {10, 8},
          {8,  6},
          {5,  6},
          {5,  11},
          {6,  12},
          {11, 12}
        };
        for (int k = 0; k < connect_line.size(); ++k) {
          auto *pose_line = render_data->add_render_annotations();
          auto *location_data_line = pose_line->mutable_line();
          location_data_line->set_normalized(true);
          location_data_line->set_x_start(key_points_of_all_parts.at(p).at(connect_line[k].first).second);
          location_data_line->set_y_start(key_points_of_all_parts.at(p).at(connect_line[k].first).first);
          location_data_line->set_x_end(key_points_of_all_parts.at(p).at(connect_line[k].second).second);
          location_data_line->set_y_end(key_points_of_all_parts.at(p).at(connect_line[k].second).first);

          pose_line->mutable_color()->set_r(0);
          pose_line->mutable_color()->set_g(255);
          pose_line->mutable_color()->set_b(255);
          pose_line->set_thickness(5);
        }
      }
    }

    for (int k = 0; k < face_bounding_boxes.size(); ++k) {
      auto *face_annotation = render_data->add_render_annotations();
      auto *location_data_rect = face_annotation->mutable_rectangle();
      location_data_rect->set_normalized(true);
      location_data_rect->set_left(face_bounding_boxes[k][0]);
      location_data_rect->set_top(face_bounding_boxes[k][1]);
      location_data_rect->set_right(face_bounding_boxes[k][0] + face_bounding_boxes[k][2]);
      location_data_rect->set_bottom(face_bounding_boxes[k][1] + face_bounding_boxes[k][3]);

      face_annotation->mutable_color()->set_r(10);
      face_annotation->mutable_color()->set_g(10);
      face_annotation->mutable_color()->set_b(250);
      face_annotation->set_thickness(3);
    }

    prev_output_timestamp_ = cc->InputTimestamp();
    cc->Outputs()
      .Tag("RENDER_DATA")
      .Add(render_data.release(), cc->InputTimestamp());
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status SyncDetectionsCalculator::Close(
    CalculatorContext* cc) {
  return ::mediapipe::OkStatus();
}
}  // namespace mediapipe

# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])  # Apache 2.0

# OpenGL utilities.

# TODO: move Compat.java in here, remove dep
# TODO: add tests
android_library(
    name = "glutil",
    srcs = glob(["**/*.java"]),
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "@com_google_code_findbugs//jar",
        "@com_google_common_flogger//jar",
        "@com_google_common_flogger_system_backend//jar",
        "@com_google_guava_android//jar",
    ],
)

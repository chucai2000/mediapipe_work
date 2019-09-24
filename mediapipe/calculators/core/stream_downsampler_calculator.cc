#include <cmath>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"

#if defined(__ANDROID__)
#include <android/log.h>
#endif  // ANDROID

namespace mediapipe {

class StreamDownsamplerCalculator : public CalculatorBase {
public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Get("", 0).SetAny();
    cc->Outputs().Get("", 0).SetSameAs(&(cc->Inputs().Get("", 0)));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    __android_log_print(
      ANDROID_LOG_INFO, "debug_yichuc", "?? StreamDownsamplerCalculator input %ld", cc->InputTimestamp().Microseconds());

    input_data_id_ = cc->Inputs().GetId("", 0);
    output_data_id_ = cc->Outputs().GetId("", 0);
    if (accum_count_ == 0) {
      __android_log_print(
        ANDROID_LOG_INFO, "debug_yichuc", "?? StreamDownsamplerCalculator output at %ld", cc->InputTimestamp().Microseconds());
      cc->Outputs().Get(output_data_id_).AddPacket(cc->Inputs().Get(input_data_id_).Value());
    }
    accum_count_ = (accum_count_ + 1) % action_every_n_;
    return ::mediapipe::OkStatus();
  }

private:
  int accum_count_ = 0;
  int action_every_n_ = 4;
  CollectionItemId input_data_id_;
  CollectionItemId output_data_id_;
};
REGISTER_CALCULATOR(StreamDownsamplerCalculator);

}  // namespace mediapipe
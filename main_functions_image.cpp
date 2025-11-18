#define PROFILE_MODEL
#define PROFILE_COUNT 5
#define INT8_MODEL

#include "constants.h"
#ifdef INT8_MODEL
#include "hello_world_int8_model_data.h"
#define MODEL_VAR g_hello_world_int8_model_data
#else
#include "hello_world_float_model_data.h"
#define MODEL_VAR g_hello_world_float_model_data
#endif
#include "main_functions.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#ifdef PROFILE_MODEL
#include "tensorflow/lite/micro/micro_profiler.h"
#endif
#include "pico/time.h"
#include "hardware/clocks.h"
#include "tensorflow/lite/micro/micro_time.h"
const int num_inputs = 3072;

#ifdef INT8_MODEL
#define ARENA_SIZE 262144
#else
#define ARENA_SIZE 65536
#endif

#define INSERT_RESOLVER(a) \
  { \
    TfLiteStatus resolve_status = a; \
    if (resolve_status != kTfLiteOk) { \
      MicroPrintf("Op resolution failed"); \
      return; \
    } \
  }

#define PAUSE(a) while(1) { sleep_ms(3000); MicroPrintf(a); }

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
#ifdef PROFILE_MODEL
tflite::MicroProfiler profiler;
#endif
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = ARENA_SIZE;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();

  // Delay so that we have time to launch the serial monitor
  static uint64_t last_action_time = time_us_64();
  const uint64_t delay_us = 5000000; // 5 second in microseconds
  while (time_us_64() - last_action_time < delay_us);
  MicroPrintf("\n\nDelay elapsed, hopefully serial port is ready now!");
  last_action_time = time_us_64();

  uint32_t cpu_freq = clock_get_hz(clk_sys);
  MicroPrintf("RP2350 CPU Clock Frequency: %u Hz", cpu_freq);
  int32_t tps = tflite::ticks_per_second();
  MicroPrintf("RP2350 ticks per second: %u", tps);
  
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  MicroPrintf("GetModel...");
  model = tflite::GetModel(MODEL_VAR);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    PAUSE("GetModel failed\n")
    return;
  }

  MicroPrintf("Initialize resolver...");
  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<6> resolver;
  INSERT_RESOLVER(resolver.AddConv2D())
  INSERT_RESOLVER(resolver.AddAdd())
  INSERT_RESOLVER(resolver.AddRelu())
  INSERT_RESOLVER(resolver.AddAveragePool2D())
  INSERT_RESOLVER(resolver.AddFullyConnected())
  INSERT_RESOLVER(resolver.AddSoftmax())

  // Build an interpreter to run the model with.
  MicroPrintf("Build Interpreter...");
#ifdef PROFILE_MODEL
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize,
						     NULL, &profiler, false);
#else
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
#endif
  
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  MicroPrintf("AllocateTensors...");
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }
    
  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;

  MicroPrintf("setup() complete.");
}

// The name of this function is important for Arduino compatibility.
void loop() {
  static int profile_count = 0;

#ifdef PROFILE_MODEL
  absolute_time_t start_time, end_time, invoke_time;
#endif  
  
#ifdef INT8_MODEL
  for(int i = 0; i < num_inputs; i++) {
    input->data.int8[i] = (char)i;
  }
#else
  for(int i = 0; i < num_inputs; i++) {
    input->data.f[i] = (float)1.0;
  }
#endif

#ifdef PROFILE_MODEL
  start_time = get_absolute_time();
#endif  
  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed at index: %d\n", inference_count);
    PAUSE("Invoke failed\n")
    return;
  }
#ifdef PROFILE_MODEL
  end_time = get_absolute_time();
  invoke_time = end_time - start_time;
  if ((profile_count % PROFILE_COUNT)==0) {  // limit number of profile events
    MicroPrintf("profile_count: %d", profile_count);
    MicroPrintf("invoke_time: %llu (usec)", invoke_time);
    profiler.Log();
  }
  profiler.ClearEvents();
  profile_count += 1;
#endif

  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  static float led_brightness = 0.0f;
  static float x = 0.0f;
  if (inference_count == 0) {
    led_brightness = (led_brightness==1.0) ? 0.0 : 1.0;  // toggle bright-dim
  }
  HandleOutput(x, led_brightness);
  
  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}

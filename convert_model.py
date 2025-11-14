import tensorflow as tf
import onnx
import onnx2tf
import numpy as np
import os
import subprocess

model = ["cnn1056000","cnn16896000","cnn2112000","cnn33792000","cnn4224000","cnn67584000","cnn8448000"]

def convert_model(name,is_int8):

    onnx_file = name+".onnx"
    tensorflow_dir = name+"_tf"
    tflite_file = ""
    tflite_cpp_file = ""
    if is_int8:
        tflite_file = name+"int8.tflite"
        tflite_cpp_file = name+"int8.cpp"
    else:
        tflite_file = name+".tflite"
        tflite_cpp_file = name+".cpp"
        
    print("onnx2tf", "-i", onnx_file, "-osd", "-o", tensorflow_dir)
    subprocess.run(["onnx2tf", "-i", onnx_file, "-osd", "-o", tensorflow_dir])

    try:
        if is_int8:
            converter_int8 = tf.lite.TFLiteConverter.from_saved_model(tensorflow_dir, signature_keys=['serving_default'])
            converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
            def representative_dataset():
                onnx_model = onnx.load(onnx_file)
                input_node = onnx_model.graph.input[0]
                input_shape = [dim.dim_value for dim in input_node.type.tensor_type.shape.dim]
                input_shape[0] = 1  # Set batch size to 1 for conversion
                for _ in range(100):
                    data = np.random.rand(*input_shape).astype(np.float32)
                    yield [data]
            converter_int8.representative_dataset = representative_dataset
            converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter_int8.inference_input_type = tf.int8
            converter_int8.inference_output_type = tf.int8
            tflite_model_int8 = converter_int8.convert()
            with open(tflite_file, "wb") as f:
                f.write(tflite_model_int8)
            print(f"TFLite model saved successfully at: {tflite_file}")
            with open(tflite_cpp_file, "w") as cppfile:
                subprocess.run(["xxd","-i",tflite_file], stdout=cppfile)
        else:
            converter = tf.lite.TFLiteConverter.from_saved_model(tensorflow_dir, signature_keys=['serving_default'])
            tflite_model = converter.convert()
            with open(tflite_file, "wb") as f:
                f.write(tflite_model)
            print(f"TFLite model saved successfully at: {tflite_file}")
            with open(tflite_cpp_file, "w") as cppfile:
                subprocess.run(["xxd","-i",tflite_file], stdout=cppfile)

    except Exception as e:
        print(f"Error converting to TFLite: {e}")

for i in range(len(model)):
    convert_model(model[i], False)
    convert_model(model[i], True)

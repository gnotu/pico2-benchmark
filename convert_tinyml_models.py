import tensorflow as tf
import onnx
import onnx2tf
import numpy as np
import os
import subprocess

model = ["ad01",
         "pretrainedResnet",
         "str_ww_ref_model",
         "vww_96"]
input_shape = [[1,640],
               [1,32,32,3],
               [1,30,1,40],
               [1,96,96,3]]
               
def convert_model_h5(name,shape):

    h5_file = name+".h5"
    tflite_file = name+".tflite"
    tflite_cpp_file = name+".cpp"

    model = tf.keras.models.load_model(h5_file)

    @tf.function
    def representative_predict(input_value):
        return model(input_value)

    # Set the input shape for the concrete function
    concrete_func = representative_predict.get_concrete_function(
        tf.TensorSpec(shape, model.inputs[0].dtype)
    )

    # 3. Instantiate the TFLite converter using the concrete function
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])




    
    #converter = tf.lite.TFLiteConverter.from_keras_model(model)
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(*shape)
            yield [data.astype(np.float32)]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()
    with open(tflite_file, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved successfully at: {tflite_file}")
    with open(tflite_cpp_file, "w") as cppfile:
        subprocess.run(["xxd","-i",tflite_file], stdout=cppfile)

def convert_model_pb(name,shape):

    tensorflow_dir = name
    tflite_file = name+".tflite"
    tflite_cpp_file = name+".cpp"

    model = tf.saved_model.load(tensorflow_dir) 
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape(shape)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(*shape)
            yield [data.astype(np.float32)]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(tflite_file, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved successfully at: {tflite_file}")
    with open(tflite_cpp_file, "w") as cppfile:
        subprocess.run(["xxd","-i",tflite_file], stdout=cppfile)

for i in range(len(model)):
    convert_model_h5(model[i], input_shape[i])

convert_model_pb("kws_ref_model",[1,49,10,1])

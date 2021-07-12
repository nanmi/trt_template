<!--
 * @Description: OpenCV-GPU TensorRT
 * @Author: nanmi
 * @Date: 2021-07-12 09:16:35
 * @LastEditTime: 2021-07-12 16:03:14
 * @LastEditors: nanmi
 * @GitHub:github.com/nanmi
 -->

# TensorRT Infer Template
This project base on [tiny-tensorrt](https://github.com/zerollzeng/tiny-tensorrt)


# News

It can speed up the whole pipeline on GPU, greatly improve the operation efficiency, and customize the pre-processing and post-processing on GPU - 2021-7-12


# Features
- [x] Preprocess in GPU
- [x] Postprocess in GPU
- [x] run whole pipeline in GPU easily
- [x] Custom onnx model output node
- [x] Engine serialization and deserialization auto
- [x] INT8 support

# System Requirements
cuda 10.0+

TensorRT 7

OpenCV 4.0+ (build with opencv-contrib module)

# Installation
Make sure you had install dependencies list above
```bash
# clone project and submodule
git clone {this repo}

cd {this repo}

mkdir build && cd build && cmake .. && make
```
use to infer tensorrt engine.

# Params

```c++
enum class BuilderFlag : int
{
    kFP16 = 0,         //!< Enable FP16 layer selection.
    kINT8 = 1,         //!< Enable Int8 layer selection.
    kDEBUG = 2,        //!< Enable debugging of layers via synchronizing after every layer.
    kGPU_FALLBACK = 3, //!< Enable layers marked to execute on GPU if layer cannot execute on DLA.
    kSTRICT_TYPES = 4, //!< Enables strict type constraints.
    kREFIT = 5,        //!< Enable building a refittable engine.
};
```

```shell
--onnx 指定onnx模型
--custom_outputs 指定模型的输出，用","隔开
--mode build engine时的数据类型，0：fp32;1:fp16;2:int8，默认值0
--engine
--batch_size 默认值1
--calibrate_data  int8模型下，指定校准数据集
--calibrate_cache int8模式下，校准表
--gpu GPU索引，默认值0
--dla 默认值-1
```

```shell
#示例
./infer_demo --onnx nanodet_m_sim.onnx --custom_outputs cls_8,cls_16,cls_32,dis_8,dis_pred_16,dis_32 --mode 1 --engine nanodet_m_sim_fp16.engine
```

# Docs

example cxx code for how to use opencv gpu version in TensorRT inference.

# About License

For the 3rd-party module and TensorRT, you need to follow their license

For the part I wrote, you can do anything you want


#pragma once
#include <NvInfer.h>

#include "plugin.hpp"
#include "serialize.hpp"

#include <cassert>

namespace {
    constexpr const char* RESIZE_BILINEAR_PLUGIN_VERSION{"001"};
    constexpr const char* RESIZE_BILINEAR_PLUGIN_NAME{"ResizeBilinear"};
}

class ResizeBilinearPlugin final : public onnx2trt::PluginV2 {
  int   _ndims;
  float _scale[nvinfer1::Dims::MAX_DIMS];
  bool _align_corners = false;
  nvinfer1::Dims _output_dims;
protected:
  void deserialize(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    deserialize_value(&serialData, &serialLength, &_ndims);
    deserialize_value(&serialData, &serialLength, &_scale);
  }
  size_t getSerializationSize() const override {
    return serialized_size(_ndims) + serialized_size(_scale) + getBaseSerializationSize();
  }
  void serialize(void *buffer) const override {
    serializeBase(buffer);
    serialize_value(&buffer, _ndims);
    serialize_value(&buffer, _scale);
  }
public:
  ResizeBilinearPlugin(std::vector<float> const& scale, bool const& align_corners)
    : _ndims(scale.size()), _align_corners(align_corners){
    assert(scale.size() <= nvinfer1::Dims::MAX_DIMS);
    std::copy(scale.begin(), scale.end(), _scale);
  }
  ResizeBilinearPlugin(void const* serialData, size_t serialLength) {
    this->deserialize(serialData, serialLength);
  }
  virtual const char* getPluginType() const override { return RESIZE_BILINEAR_PLUGIN_NAME; }

  virtual void destroy() override { delete this; }

  virtual nvinfer1::IPluginV2* clone() const override { return new ResizeBilinearPlugin{std::vector<float>(_scale, _scale + _ndims), false}; }

  virtual const char* getPluginVersion() const override { return RESIZE_BILINEAR_PLUGIN_VERSION; }

  virtual void setPluginNamespace(const char* pluginNamespace) override {}

  virtual const char* getPluginNamespace() const override { return ""; }

  virtual int getNbOutputs() const override { return 1; }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputs, int nbInputDims) override;
  virtual int initialize() override;

  float area_pixel_compute_scale(int input_size, int output_size);

  int enqueue(int batchSize,
              const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;
};

class ResizeBilinearPluginCreator : public nvinfer1::IPluginCreator
{
public:
  ResizeBilinearPluginCreator() {}

  ~ResizeBilinearPluginCreator() {}

  const char* getPluginName() const { return RESIZE_BILINEAR_PLUGIN_NAME; }

  const char* getPluginVersion() const { return RESIZE_BILINEAR_PLUGIN_VERSION; }

  const nvinfer1::PluginFieldCollection* getFieldNames() { std::cerr<< "Function not implemented" << std::endl; return nullptr; }

  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) { std::cerr<< "Function not implemented" << std::endl; return nullptr; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) { return new ResizeBilinearPlugin{serialData, serialLength}; }

  void setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }

  const char* getPluginNamespace() const { return mNamespace.c_str(); }
private:
    std::string mNamespace;
};

REGISTER_TENSORRT_PLUGIN(ResizeBilinearPluginCreator);

#include <cuda_fp16.h>
#include <cassert>
#include <algorithm>
#include "ResizeBilinear.hpp"

// TODO: Move this to a common header
inline bool is_CHW(nvinfer1::Dims const& dims) {
  return (dims.nbDims == 3 &&
          dims.type[0] == nvinfer1::DimensionType::kCHANNEL &&
          dims.type[1] == nvinfer1::DimensionType::kSPATIAL &&
          dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}

nvinfer1::Dims ResizeBilinearPlugin::getOutputDimensions(int index,
                                                        const nvinfer1::Dims *inputDims,
                                                        int nbInputs) {
  assert(nbInputs == 1);
  nvinfer1::Dims const& input = inputDims[0];
  assert(is_CHW(input));
  assert(_ndims == 2);
  assert(index == 0);
  nvinfer1::Dims output;
  output.nbDims = input.nbDims;
  int s = 0;
  for( int d=0; d<input.nbDims; ++d ) {
    output.type[d] = input.type[d];
    if( input.type[d] == nvinfer1::DimensionType::kSPATIAL ) {
      output.d[d] = int(input.d[d] * _scale[s++]);
    } else {
      output.d[d] = input.d[d];
    }
  }
  return output;
}

int ResizeBilinearPlugin::initialize() {
  _output_dims = this->getOutputDimensions(0, &this->getInputDims(0), 1);
  assert(is_CHW(this->getInputDims(0)));
  assert(is_CHW(_output_dims));
  assert(_ndims == 2);
  return 0;
}

__device__
void area_pixel_compute_source_index(float &rc,
                                     float scale,
                                     int dst_index,
                                     bool align_corners,
                                     bool cubic = false)
{
    if (align_corners)
    {
        rc = scale * dst_index;
        return;
    }
    else
    {
        float src_idx = scale * (dst_index + 0.5) - 0.5;
        rc = (!cubic && src_idx < 0) ? float(0.0) : src_idx;
        return;
    }
}

template <typename Data>
__global__
void resize_bilinear_kernel_2d(int n,
                               int batchsize,
                               int channels,
                               int height1,
                               int width1,
                               int height2,
                               int width2,
                               float rheight,
                               float rwidth,
                               bool align_corners,
                               Data const* idata,
                               Data*       odata) 
{
    const int in_batchsize_stride = channels * height1 * width1;
    const int in_channels_stride = height1 * width1;
    const int out_batchsize_stride = channels * height2 * width2;
    const int out_channels_stride = height2 * width2;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
    {
        const int w2 = index % width2;
        const int h2 = index / width2;
        if (height1 == height2 && width1 == width2)
        {
            const int h1 = h2;
            const int w1 = w2;
            for (int n = 0; n < batchsize; n++)
            {
                for (int c = 0; c < channels; c++)
                {
                    odata[n * out_batchsize_stride + c * out_channels_stride + h2 * width2 + w2]
                        = idata[n * in_batchsize_stride + c * in_channels_stride + h1 * width1 + w1];
                }
            }
            return;
        }
        //
        float h1r;
        area_pixel_compute_source_index(h1r, rheight, h2, align_corners, /*cubic=*/false);
        const int h1 = h1r;
        const int h1p = (h1 < height1 - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = static_cast<float>(1) - h1lambda;
        //
        float w1r;
        area_pixel_compute_source_index(w1r, rwidth, w2, align_corners, /*cubic=*/false);
        const int w1 = w1r;
        const int w1p = (w1 < width1 - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = static_cast<float>(1) - w1lambda;
        //
        for (int n = 0; n < batchsize; n++)
        {
            for (int c = 0; c < channels; ++c)
            {
                const float val = 
                    h0lambda * 
                    (w0lambda * idata[n * in_batchsize_stride + c * in_channels_stride + h1 * width1 + w1] +
                     w1lambda * idata[n * in_batchsize_stride + c * in_channels_stride + h1 * width1 + (w1 + w1p)]) +
                    h1lambda *
                    (w0lambda * idata[n * in_batchsize_stride + c * in_channels_stride + (h1 + h1p) * width1 + w1] +
                     w1lambda * idata[n * in_batchsize_stride + c * in_channels_stride + (h1 + h1p) * width1 + (w1 + w1p)]);
                odata[n * out_batchsize_stride + c * out_channels_stride + h2 * width2 + w2] = val;
            }
        }
    }
}

float ResizeBilinearPlugin::area_pixel_compute_scale(int input_size,
                                                   int output_size)
{
    if(output_size > 1)
    {
        return _align_corners ? float(input_size - 1) / (output_size - 1) : float(input_size) / output_size;
    }
    else
    {
        return 0.0;
    }
}

int ResizeBilinearPlugin::enqueue(int batchSize,
                                 const void *const *inputs, void **outputs,
                                 void *workspace, cudaStream_t stream)
{
    auto const& input_dims = this->getInputDims(0);
    switch( _ndims )
    {
        case 2:
            {
                const int channels = input_dims.d[0];
                const int input_height = input_dims.d[1];
                const int input_width = input_dims.d[2];
                const int output_height = _output_dims.d[1];
                const int output_width = _output_dims.d[2];
                int obatchstride = _output_dims.d[1] * _output_dims.d[2];
                int num_kernels = obatchstride;
                int num_threads = 512;
                int blocks = int((num_kernels + num_threads - 1) / num_threads);
                int grid = num_threads;
                float rheight = area_pixel_compute_scale(input_height, output_height);
                float rwidth  = area_pixel_compute_scale(input_width, output_width);

                resize_bilinear_kernel_2d<<<blocks, grid, 0, stream>>>(
                        num_kernels, batchSize, channels, input_height, input_width,
                        output_height, output_width, rheight, rwidth, _align_corners,
                        static_cast<float const*>( inputs[0]), static_cast<float*>(outputs[0]));
                return cudaGetLastError() != cudaSuccess;
            }
        default: return -1;
    }
}

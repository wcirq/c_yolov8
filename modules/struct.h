#ifndef STRUCT_H
#define STRUCT_H
#include <sys/types.h>
#include <stddef.h>

typedef struct {
  int layerNum;        // transformer dimension
} Config;

typedef struct
{
    int size;   // tensoe size
    int *shape;  // 固定为4维,依次为 (batch channel height width) 默认为0
    float *data;
}Tensor;


typedef struct {
    Tensor conv_weight;        // shape 48 3 3 3
    Tensor conv_bias;          // shape 48
    Tensor bn_weight;          // shape 48
    Tensor bn_bias;            // shape 48
    Tensor bn_running_var;     // shape 48
    Tensor bn_running_mean;    // shape 48
    float bn_eps;              // value eg:0.0001
} ConvArgument;


typedef struct
{
    ConvArgument conv1;
}Layers;


typedef struct {
  // current wave of activations
  Tensor x;      // activation at current time stamp (dim,)
  Tensor y;      // 
  Tensor logits; // output logits
} RunState;


typedef struct
{
    Config config;
    Layers layers;
    RunState state;             // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd;            // file descriptor for memory mapping
    float *data;       // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
}Model;

#endif
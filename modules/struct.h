

typedef struct {
  int layerNum;        // transformer dimension
} Config;

typedef struct
{
    int *size;   // tensoe size
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
    float *bn_eps;              // value eg:0.0001
} Conv;


typedef struct
{
    Conv conv1;
}Layers;


typedef struct
{
    Config config;
    Layers layers;
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd;            // file descriptor for memory mapping
    float *data;       // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
}Model;

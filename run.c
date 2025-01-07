#include "modules/layers.h"
#include "modules/struct.h"


int memory_map_tensor(Tensor *tensor, float *ptr) {
    unsigned long long size_len = 1;
    unsigned long long shape_len = 4;
    tensor->size = ptr;
    ptr+=size_len;
    tensor->shape = ptr;
    ptr+=shape_len;
    tensor->data = ptr;
    return *tensor->size + size_len + shape_len;
}


void memory_map_weights(Model *model, Config *config, float *ptr) {
    int layerNum = config->layerNum;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    Layers layers = model->layers;

    // embed_tokens
    Conv conv1 = layers.conv1;
    int size = memory_map_tensor(&conv1.conv_weight, ptr);
    ptr += size;
    size = memory_map_tensor(&conv1.conv_bias, ptr);
    ptr += size;
    size = memory_map_tensor(&conv1.bn_weight, ptr);
    ptr += size;
    size = memory_map_tensor(&conv1.bn_bias, ptr);
    ptr += size;
    size = memory_map_tensor(&conv1.bn_running_var, ptr);
    ptr += size;
    size = memory_map_tensor(&conv1.bn_running_mean, ptr);
    ptr += size;
    conv1.bn_eps = ptr;
}


void read_checkpoint(char *checkpoint, Config *config, Model *weights, int *fd, float **data, ssize_t *file_size) {
  FILE *file = fopen(checkpoint, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s, it may not exist.\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  // read in the config header
  if (fread(config, sizeof(Config), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  // figure out the file size
#if defined _WIN32
  _fseeki64(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = _ftelli64(file); // get the file size, in bytes
#else
  fseek(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = ftell(file); // get the file size, in bytes
#endif
  fclose(file);
  // memory map the Transformer weights into the data pointer
  *fd = open(checkpoint, O_RDONLY); // open in read only mode
  if (*fd == -1) {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }
  *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }
  float *weights_ptr = *data + sizeof(Config) / sizeof(float);
  memory_map_weights(weights, config, weights_ptr);
}



void build_transformer(Model *model, char *checkpoint_path){
  // read in the Config and the Weights from the checkpoint
  read_checkpoint(checkpoint_path, &model->config, &model->layers, &model->fd, &model->data, &model->file_size);
}


int main(int argc, char *argv[])
{
    char *checkpoint_path = (argc > 1) ? argv[1] : "yolov8.bin";
    Model model;
    build_transformer(&model, checkpoint_path);
    printf("");
}
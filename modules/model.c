#include "model.h"



Tensor forward(Model *model, Tensor *x) 
{
  // a few convenience variables
  Config config = model->config;
  Layers layers = model->layers;
  RunState s = model->state;

  // first layer
  fuse_conv_bn(&x, &layers, &s.y);
  
  return s.logits;
}
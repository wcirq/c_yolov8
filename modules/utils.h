#ifndef UTILS_H
#define UTILS_H
#include "struct.h"

void initTensor(Tensor *t, int b, int c, int h, int w);

void free_model(Model *model);

void free_tensor(Tensor *tensor);

#endif
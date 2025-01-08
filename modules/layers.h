#ifndef LAYERS_H
#define LAYERS_H
#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdbool.h>
#include "struct.h"


void matmul(Tensor *x, Tensor *weights, Tensor *y);

void bn(Tensor *x, Tensor *weights, Tensor *y);

void conv(Tensor *x, Tensor *weights, Tensor *y);

void fuse_conv_bn(Tensor *x, ConvArgument *convWeights, Tensor *y);

#endif
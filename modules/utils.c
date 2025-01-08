#include "utils.h"

void initTensor(Tensor *t, int b, int c, int h, int w)
{
    int size = b*c*h*w;
    t->size = size;
    t->shape = (int *)malloc(4 * sizeof(int *));
    t->shape[0] = b;
    t->shape[1] = c;
    t->shape[2] = h;
    t->shape[3] = w;
    t->data = (float *)malloc(t->size * sizeof(float *));
}

void free_model(Model *model) {
     
}

void free_tensor(Tensor *tensor){

}
// llama_interface.h
#ifndef LLAMA_INTERFACE_H
#define LLAMA_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

void llama_initialize(int num_inputs, int num_outputs, int num_layers, int num_neurons);
void llama_train(double* inputs, double* outputs, int num_samples);
double llama_predict(double* input);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_INTERFACE_H

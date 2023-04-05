#include "llama.h"
#include "llama_interface.h"
#include "llama.cpp"

static struct llama_context * ml;

extern "C" {
    void llama_initialize(int num_inputs, int num_outputs, int num_layers, int num_neurons) {
        struct llama_context_params params = llama_context_default_params();
        ml = llama_init_from_file("models/ggml-vocab.bin", params); // Remplacez "model_file_path" par le chemin du fichier de modèle LlaMa
    }

    void llama_train(double* vader_sentiments, double* trading_signals, int num_samples) {
        for (int i = 0; i < num_samples; ++i) {
            llama_train(ml, vader_sentiments + i * ml->params.num_inputs, trading_signals + i * ml->params.num_outputs);
        }
    }

    double llama_predict(double* vader_sentiment) {
        double prediction;
        llama_predict(ml, vader_sentiment, &prediction);
        return prediction;
    }
}

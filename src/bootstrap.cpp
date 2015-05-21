#include <random>
#include "bootstrap.h"

typedef std::mt19937 RandomNumberGenerator;
using std::vector;
using std::shared_ptr;

// Returns indexes to be used for uniform sampling with replacement
vector<size_t> sample_with_replacement(RandomNumberGenerator& rng, size_t vector_size, size_t sample_size) {
    std::uniform_int_distribution<int> uniform(0, vector_size - 1);
    std::vector<size_t> result(sample_size);
    for (size_t i = 0; i < sample_size; i++) {
        result[i] = uniform(rng);
    }
    return result;
}

// Sample a vector by a vector of indexes
template<typename T>
vector<T> sample(const vector<T>& input, vector<size_t> indexes) {
    vector<T> output(indexes.size());
    for (size_t i = 0; i < indexes.size(); i++) {
        output[i] = input[indexes[i]];
    }
    return output;
}

void Bootstrap::solve() {
    // Uniform distribution for sampling observations
    std::random_device rd;
    RandomNumberGenerator rng(rd());

    // Bootstrap samples
    vector<shared_ptr<Model>> bootstrap_models(number_of_samples);

    for (size_t i = 0; i < number_of_samples; i++) {
        // Deep-copy base model using 'virtual constructor' idiom
        shared_ptr<Model> bs_sample(base_model->clone());
        bs_sample->features.reset(new FeaturesGraph(*bs_sample->features));

        // Only consider first edge for now, this should be extended
        obs_pair& edge = bs_sample->features->edges[0];

        // Bootstrap and solve it
        vector<size_t> indexes = sample_with_replacement(rng, edge.obs_a.size(), size_of_samples);
        bs_sample->features->number_of_matches = size_of_samples;
        edge.obs_a = sample(edge.obs_a, indexes);
        edge.obs_b = sample(edge.obs_b, indexes);
        bs_sample->solve();
        bootstrap_models[i] = bs_sample;
    }

    // Compute the bootstrap means and stds
    // or just export data to make a density map (distribution of results)
    for (auto const& model : bootstrap_models) {
        for (auto& ext : model->final_external()) {
            for (auto const& item : ext) {
                std::cout << item << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

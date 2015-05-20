#include <random>
#include "bootstrap.h"

typedef std::mt19937 RandomNumberGenerator;
using std::vector;

// Returns indexes to be used for uniform sampling with replacement
vector<size_t> sample_with_replacement(RandomNumberGenerator& rgn, size_t vector_size) {
    std::uniform_int_distribution<int> uniform(0, vector_size - 1);
    std::vector<size_t> result(vector_size);
    for (size_t i = 0; i < vector_size; i++) {
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

    // Create number_of_samples "bootstrap samples"
    vector<std::shared_ptr<Model>> bootstrap_models;

    for (size_t i = 0; i < number_of_samples; i++) {
        // Copy base model (make sure to copy correctly through pointer to base)
        std::shared_ptr bs_sample(new );

        // Modify features to be a bootstrapped sample of original features
        // Solve
    }

    // Compute the bootstrap means and stds
    // or just export data to make a density map (distribution of results)
}

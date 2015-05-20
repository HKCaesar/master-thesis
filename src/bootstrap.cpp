#include <random>
#include "bootstrap.h"

typedef std::mt19937 RandomNumberGenerator;

void Bootstrap::solve() {
    std::random_device rd;
    RandomNumberGenerator rng;
    rng.seed(rd());
    std::uniform_int_distribution<int> uniform(0, 100);

    std::cout << uniform(rng) << std::endl;
    std::cout << uniform(rng) << std::endl;
    std::cout << uniform(rng) << std::endl;
    std::cout << uniform(rng) << std::endl;
    std::cout << uniform(rng) << std::endl;

    // Create number_of_samples "bootstrap samples"
    // with replacement
    //
    // Solve each sample
    //
    // Compute the bootstrap means and stds
    // or just export data to make a density map (distribution of results)
}

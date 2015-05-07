#ifndef MODEL_H
#define MODEL_H

#include <utility>
#include "ceres/ceres.h"
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include "internal.h"

// Base class for models' cost functions
// Provides a factory to hide the construction of the autodiff'ed CostFunction object
template <typename CostFunctor,
       int kNumResiduals,  // Number of residuals, or ceres::DYNAMIC.
       int N0,       // Number of parameters in block 0.
       int N1 = 0,   // Number of parameters in block 1.
       int N2 = 0,   // Number of parameters in block 2.
       int N3 = 0,   // Number of parameters in block 3.
       int N4 = 0,   // Number of parameters in block 4.
       int N5 = 0,   // Number of parameters in block 5.
       int N6 = 0,   // Number of parameters in block 6.
       int N7 = 0,   // Number of parameters in block 7.
       int N8 = 0,   // Number of parameters in block 8.
       int N9 = 0>   // Number of parameters in block 9.
struct CostFunction {
    template <typename... Args>
    static ceres::CostFunction* make(Args&&... args) {
        return new ceres::AutoDiffCostFunction<CostFunctor, kNumResiduals, N0, N1, N2, N3, N4, N5, N6, N7, N8, N9>
            (new CostFunctor(std::forward<Args>(args)...));
    }
};

// Utility class to enable logging of solutions at each sovler step
template <typename SolutionType>
class LogSolutionCallback : public ceres::IterationCallback {
    std::vector<SolutionType>& solutions;
    const SolutionType& working_solution;
public:
    LogSolutionCallback(std::vector<SolutionType>& v, const SolutionType& sol) : solutions(v), working_solution(sol) {}
    virtual ~LogSolutionCallback() {}
    virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary&) {
        solutions.push_back(working_solution);
        return ceres::SOLVER_CONTINUE;
    }
};

struct UnprovidedFinal : public std::runtime_error {
    UnprovidedFinal(const std::string& str) : std::runtime_error(std::string("UnprovidedFinal: ") + str) {}
};

// Base class for models
// Must overwrite solve()
class Model {
public:

    Model() : solved(false) {
        // Solver options common to all models
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.max_linear_solver_iterations = 3;
        options.max_num_iterations = 30;
        options.num_threads = 1;
    }

    virtual void solve() = 0;
    virtual ~Model() {}

    // Optional interfaces for initialising another model with this as parent
    // Should return the best known value after optimization
    virtual std::vector<std::array<double, 6>> final_external() const { throw UnprovidedFinal("final_external"); }
    virtual internal_t final_internal() const { throw UnprovidedFinal("final_internal"); }

    // Enable logging of solutions at every step
    template <typename T>
    void enable_logging(std::vector<T>& solutions, const T& working_solution) {
        options.update_state_every_iteration = true;
        solution_logger.reset(new LogSolutionCallback<T>(solutions, working_solution));
        options.callbacks.push_back(solution_logger.get());
    }

    template <class Archive>
    void serialize(Archive& ar) {
        ar(cereal::make_nvp("solved", solved),
           cereal::make_nvp("features", features));
    }

    bool solved;
    std::shared_ptr<FeaturesGraph> features;
    std::unique_ptr<ceres::IterationCallback> solution_logger;
    ceres::Problem problem;
    ceres::Solver::Options options;
};

#endif

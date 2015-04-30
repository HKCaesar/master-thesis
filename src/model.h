#ifndef MODEL_H
#define MODEL_H

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/json.hpp>

class Model {
public:
    virtual void solve() = 0;
    virtual ~Model() {}
};

template <typename ModelType, typename SolutionType>
class LogSolutionCallback : public ceres::IterationCallback {
    ModelType& model;
    const SolutionType& working_solution;
public:
    LogSolutionCallback(ModelType& m, const SolutionType& w) : model(m), working_solution(w) {}
    virtual ~LogSolutionCallback() {}
    virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary&) {
        model.solutions.push_back(working_solution);
        return ceres::SOLVER_CONTINUE;
    }
};


#endif

#ifndef MODEL_H
#define MODEL_H

#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>

class Model {
public:
    virtual void solve() = 0;
    virtual ~Model() {}

    template <class Archive>
    void serialize(Archive& ar) {
        ar(cereal::make_nvp("features", features));
    }

    std::shared_ptr<FeaturesGraph> features;
};

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

#endif

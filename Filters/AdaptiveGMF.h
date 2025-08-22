
#pragma once

#include <vector>

#include "Filters/GaussianComponent.h"
#include "Config/ConfigParser.h"
#include "Utils/Logger.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class AdaptiveGMF {
public:
    AdaptiveGMF(const FilterParameters& filter_params,
                const ModelParameters& model_params,
                Logger& logger);
    ~AdaptiveGMF();

    void predict();
    void update(double observed_return);

    double getEstimatedAnnualVolatility() const;
    int getNumComponents() const;

private:
    std::vector<GaussianComponent> components;
    FilterParameters filter_params;
    ModelParameters model_params;
    Logger& logger;

    void pruneComponents();
    void mergeComponents();
    void splitComponents();
};

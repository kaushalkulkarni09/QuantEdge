// Data/DataSimulator.h
#pragma once

#include <vector>
#include <string>
// Custom headers (paths relative to project root)
#include "Models/SVJumpLeverageParameters.h" // Model parameters struct
#include "Config/ConfigParser.h"             // For SimulationParameters
#include "Utils/Logger.h"                    // Logger class

class DataSimulator {
public:
    DataSimulator(const SimulationParameters& sim_params,
                  const ModelParameters& model_params,
                  Logger& logger);

    std::vector<double> generateData();
    std::vector<double> getTrueVolatilities() const;

private:
    SimulationParameters sim_params;
    ModelParameters model_params;
    Logger& logger;
    std::vector<double> true_volatilities;

    double generateStandardNormal();
    double generatePoisson(double lambda);
    double generateNormal(double mean, double std_dev);
};

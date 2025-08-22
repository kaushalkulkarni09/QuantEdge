// Data/DataSimulator.cpp
#include "Data/DataSimulator.h" // Self-include
#include <random>
#include <cmath>
#include <limits> // For std::numeric_limits


static std::random_device rd;
static std::mt19937 generator(rd());
static std::normal_distribution<> normal_dist(0.0, 1.0); // Standard normal for Wiener processes
static std::poisson_distribution<> poisson_dist(0.0); // Will set lambda later for jumps

DataSimulator::DataSimulator(const SimulationParameters& sim_params,
                             const ModelParameters& model_params,
                             Logger& logger)
    : sim_params(sim_params), model_params(model_params), logger(logger) {}

double DataSimulator::generateStandardNormal() {
    return normal_dist(generator);
}

double DataSimulator::generatePoisson(double lambda) {

    poisson_dist.param(std::poisson_distribution<>::param_type(lambda)); // Update lambda
    return poisson_dist(generator);
}

double DataSimulator::generateNormal(double mean, double std_dev) {
    // Generate a standard normal and scale/shift it
    return mean + std_dev * generateStandardNormal();
}

std::vector<double> DataSimulator::generateData() {
    std::vector<double> simulated_returns;
    true_volatilities.clear(); // Clear previous simulation data

    // Time step (daily)
    double dt = 1.0 / sim_params.NumTradingDaysPerYear;


    double initial_daily_variance = sim_params.InitialTrueVolSq / sim_params.NumTradingDaysPerYear;
    double current_log_vol_sq = std::log(initial_daily_variance);
    // --- CRITICAL CHANGE END ---

    double current_asset_price = 100.0; // Start with an arbitrary price, returns are relative

    int total_steps = sim_params.NumYears * sim_params.NumTradingDaysPerYear;

    logger.log(Logger::INFO, "Generating %d steps of simulated data...", total_steps);

    for (int i = 0; i < total_steps; ++i) {
        // Generate correlated Brownian motions for volatility and asset price
        // dW1 and dW_correlated are standard normal, then scaled by sqrt(dt) for the SDE.
        double dW1_std_norm = generateStandardNormal(); // Raw N(0,1) for volatility process
        double dW_uncorrelated_std_norm = generateStandardNormal(); // Raw N(0,1) for uncorrelated part


        double dZ1 = generateStandardNormal(); // This is the standard normal increment for volatility (d_log_v_t)
        double dZ2_uncorr = generateStandardNormal(); // Uncorrelated standard normal increment

        // The Wiener processes are dW_t = Z * sqrt(dt)
        // The correlation is between the underlying standard normal variables (Z1 and Z2)
        // So, Z2 = Rho * Z1 + sqrt(1-Rho^2) * Z2_uncorr
        double dZ2 = model_params.Rho * dZ1 + std::sqrt(1.0 - model_params.Rho * model_params.Rho) * dZ2_uncorr;

        // Volatility process (Ornstein-Uhlenbeck for log_v_t)
        // d(log_v_t) = Kappa * (Theta - log_v_t) dt + SigmaV * dW1
        // dW1 = dZ1 * sqrt(dt)
        current_log_vol_sq += model_params.Kappa * (model_params.Theta - current_log_vol_sq) * dt +
                              model_params.SigmaV * dZ1 * std::sqrt(dt); // Explicitly scale dZ1 by sqrt(dt)


        double current_vol_sq = std::exp(current_log_vol_sq);

        if (current_vol_sq < std::numeric_limits<double>::epsilon()) {
            current_vol_sq = std::numeric_limits<double>::epsilon();
            current_log_vol_sq = std::log(current_vol_sq);
        }


        // Jump process
        int num_jumps = static_cast<int>(generatePoisson(model_params.Lambda * dt)); // Number of jumps in dt
        double jump_component = 0.0;
        for (int j = 0; j < num_jumps; ++j) {
            jump_component += generateNormal(model_params.MuJ, model_params.SigmaJ); // Sum of jump sizes
        }

        // Asset price process (log returns)
        // d(log S)_t = (RiskFreeRate - 0.5 * v_t^2) dt + sqrt(v_t^2) * dW2_t + dJ_t
        // Here, v_t^2 is current_vol_sq
        // dW2_t = dZ2 * sqrt(dt)
        double expected_return_comp = (model_params.RiskFreeRate - 0.5 * current_vol_sq) * dt;
        double stochastic_return_comp = std::sqrt(current_vol_sq) * dZ2 * std::sqrt(dt); // Explicitly scale dZ2 by sqrt(dt)

        double daily_log_return = expected_return_comp + stochastic_return_comp + jump_component;

        simulated_returns.push_back(daily_log_return);

        // Update asset price (for conceptual tracking, not strictly needed for filter input)
        current_asset_price *= std::exp(daily_log_return);


        true_volatilities.push_back(std::sqrt(current_vol_sq * sim_params.NumTradingDaysPerYear));
    }

    logger.log(Logger::INFO, "Simulated data generation complete.");
    return simulated_returns;
}

std::vector<double> DataSimulator::getTrueVolatilities() const {
    return true_volatilities;
}
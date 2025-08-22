// Filters/AdaptiveGMF.cpp
// Implementation of the AdaptiveGMF (Adaptive Gaussian Mixture Filter) class.

#include "Filters/AdaptiveGMF.h" // Self-include
#include <iostream>
#include <limits>    // For numeric_limits
#include <cmath>     // For std::log, std::sqrt, std::exp, M_PI
#include <algorithm> // For std::remove_if, std::max_element, std::max, std::sort
#include <numeric>   // For std::accumulate (if used for sum)
#include <Eigen/Cholesky> // For Cholesky decomposition (if needed for PD checks)

// --- CRITICAL NUMERICAL STABILITY CONSTANT ---

const double small_positive_val = 1e-12; // Adjusted for better stability with very small values
// ---

// Constructor
AdaptiveGMF::AdaptiveGMF(const FilterParameters& filter_params,
                         const ModelParameters& model_params,
                         Logger& logger)
    : filter_params(filter_params), model_params(model_params), logger(logger) {
    // Initialize filter state with Gaussian components
    Eigen::VectorXd initial_mean(1);


    double initial_daily_variance_filter = filter_params.InitialFilterVolSq / model_params.DaysPerYear;
    // Add a check to prevent log(0) or log(negative)
    if (initial_daily_variance_filter <= 0) {
        logger.log(Logger::LogLevel::ERROR, "Initial daily variance for filter is non-positive (%f). Using small_positive_val (%e).", initial_daily_variance_filter, small_positive_val);
        initial_daily_variance_filter = small_positive_val; // Use the defined small positive number for safety
    }
    initial_mean << std::log(initial_daily_variance_filter);

    Eigen::MatrixXd initial_covariance(1, 1);
    initial_covariance << filter_params.InitialFilterCov;

    // Ensure initial_covariance is positive definite and not too small
    if (initial_covariance.determinant() <= 0 || initial_covariance(0,0) < small_positive_val) {
        initial_covariance(0,0) = std::max(initial_covariance(0,0), small_positive_val); // Cap at small_positive_val
        if (initial_covariance.determinant() <= 0) { // If still non-positive after capping, add epsilon
             initial_covariance(0,0) += std::numeric_limits<double>::epsilon();
        }
        logger.log(Logger::LogLevel::WARNING, "Initial covariance adjusted to %e to ensure positivity/minimum value.", initial_covariance(0,0));
    }

    for (int i = 0; i < filter_params.NumInitialComponents; ++i) {
        // For multiple initial components, they are initialized identically as specified.
        components.emplace_back(initial_mean, initial_covariance, 1.0 / filter_params.NumInitialComponents);
    }
    logger.log(Logger::LogLevel::INFO, "Adaptive GMF initialized with %d components.", filter_params.NumInitialComponents);
}

AdaptiveGMF::~AdaptiveGMF() {
    // Python interpreter finalization is handled by PythonDataFetcher's destructor if used.
}

// Predict step (Time Update) - Corrected for Ornstein-Uhlenbeck process
void AdaptiveGMF::predict() {
    double dt = 1.0 / model_params.DaysPerYear; // Daily step size for SDE discretization

    for (auto& comp : components) {
        // Stochastic Volatility SDE (Ornstein-Uhlenbeck process for log_v_t):
        // d(log_v_t) = kappa * (theta - log_v_t) dt + sigma_v * dW1

        // EKF Prediction for Mean (mu_prime):
        // For OU process: mu_t+dt = Theta + exp(-Kappa*dt) * (mu_t - Theta)
        double prev_mu = comp.mean(0);
        double predicted_mu = model_params.Theta + std::exp(-model_params.Kappa * dt) * (prev_mu - model_params.Theta);
        comp.mean(0) = predicted_mu; // Update the mean of the component

        // EKF Prediction for Covariance (P_prime):
        // For OU process: P(t+dt) = F * P(t) * F^T + Q
        // F = exp(-Kappa*dt) for 1D
        // Q = SigmaV^2 * (1 - exp(-2*Kappa*dt)) / (2*Kappa)
        double F_factor = std::exp(-model_params.Kappa * dt);
        double Q_process_noise;
        // Handle Kappa = 0 case (becomes a random walk with drift)
        if (model_params.Kappa == 0.0) {
            Q_process_noise = model_params.SigmaV * model_params.SigmaV * dt; // Random walk process noise
        } else {
            Q_process_noise = (model_params.SigmaV * model_params.SigmaV) * (1.0 - std::exp(-2.0 * model_params.Kappa * dt)) / (2.0 * model_params.Kappa);
        }

        double prev_cov = comp.covariance(0, 0);
        double predicted_cov = F_factor * prev_cov * F_factor + Q_process_noise; // P_prime = F P F^T + Q (F^T is just F for 1D scalar)

        // Ensure predicted covariance remains positive and not too small
        comp.covariance(0, 0) = std::max(predicted_cov, small_positive_val);

        // Log predictions (optional, for detailed debugging)
        logger.log(Logger::LogLevel::DEBUG, "Predict: Comp Mean: %e -> %e, Cov: %e -> %e", prev_mu, comp.mean(0), prev_cov, comp.covariance(0,0));
    }
}

// Update step (Measurement Update)
void AdaptiveGMF::update(double observed_return) {
    double sum_weighted_likelihoods = 0.0;
    // double dt = 1.0 / model_params.DaysPerYear; // dt is not directly used in h_x for daily volatility

    // Squared observed return for the EKF observation
    double observed_return_sq = observed_return * observed_return;

    // Log the observed return and its square
    logger.log(Logger::LogLevel::DEBUG, "--- Update step for return: %f, squared_return: %f ---", observed_return, observed_return_sq);

    // Store likelihoods for normalization later
    std::vector<double> current_likelihoods;
    current_likelihoods.reserve(components.size());

    // Iterate through each component for the update
    for (size_t i = 0; i < components.size(); ++i) {
        GaussianComponent& comp = components[i];

        // Log component details before update
        logger.log(Logger::LogLevel::DEBUG, "  Component %zu (before update): Mean[0]=%f, Cov[0,0]=%f, Weight=%f",
                   i, comp.mean(0), comp.covariance(0,0), comp.weight);

        // Predicted state mean (log_vol_sq) and covariance from the predict step
        double predicted_log_vol_sq_mean = comp.mean(0);
        double predicted_log_vol_sq_cov = comp.covariance(0, 0);


        double h_x = std::exp(predicted_log_vol_sq_mean);

        // Jacobian H_t = d(h_x) / d(predicted_log_vol_sq_mean)
        // d(exp(x))/dx = exp(x)
        double H_t = h_x; // The derivative of exp(x) with respect to x is itself.

        // Measurement noise covariance R_t for the *squared* return
        // For a zero-mean Gaussian variable Z ~ N(0, V), E[Z^2] = V. Var(Z^2) = 2*V^2.
        // Here, V = h_x.
        double R_t = 2.0 * h_x * h_x; // Var(r_t^2) = 2 * (E[r_t^2])^2 if returns are Gaussian.

        // Log calculated EKF terms before checks
        logger.log(Logger::LogLevel::DEBUG, "    h_x (exp_sq_ret): %e, H_t (Jacobian): %e, R_t (meas_noise_cov): %e", h_x, H_t, R_t); // Use %e for scientific notation

        // --- CRITICAL CAPPING FOR NUMERICAL STABILITY ---
        // Ensure terms are not too small to prevent division by zero or numerical underflow
        if (h_x < small_positive_val) {
            logger.log(Logger::LogLevel::WARNING, "    h_x was too small (%e). Capping to %e.", h_x, small_positive_val);
            h_x = small_positive_val;
            // H_t depends on h_x, so it is also capped implicitly.
        }

        if (R_t < small_positive_val) {
             logger.log(Logger::LogLevel::WARNING, "    R_t was too small (%e). Capping to %e.", R_t, small_positive_val);
             R_t = small_positive_val;
        }

        // Kalman Filter Equations
        double innovation = observed_return_sq - h_x; // Innovation: (observed squared return - expected squared return)

        double S_t = H_t * predicted_log_vol_sq_cov * H_t + R_t; // Innovation covariance

        // Ensure S_t is positive before division. Add an absolute floor.
        if (S_t < small_positive_val) {
            logger.log(Logger::LogLevel::WARNING, "    S_t was too small (%e). Capping to %e.", S_t, small_positive_val);
            S_t = small_positive_val;
        }

        double K_t = predicted_log_vol_sq_cov * H_t / S_t; // Kalman gain

        // Log innovation, S_t, K_t
        logger.log(Logger::LogLevel::DEBUG, "    Innovation: %e, S_t (innov_cov): %e, K_t (Kalman_gain): %e", innovation, S_t, K_t);


        // Update mean and covariance
        comp.mean(0) += K_t * innovation;
        comp.covariance(0, 0) -= K_t * H_t * predicted_log_vol_sq_cov;

        // Ensure covariance remains positive and not too small
        comp.covariance(0,0) = std::max(comp.covariance(0,0), small_positive_val);
        if (comp.covariance(0,0) < std::numeric_limits<double>::epsilon()) { // Using standard epsilon here as an additional safety if small_positive_val is not enough
            logger.log(Logger::LogLevel::WARNING, "    Component covariance became non-positive (%f). Capping to epsilon.", comp.covariance(0,0));
            comp.covariance(0,0) = std::numeric_limits<double>::epsilon();
        }


        // Log component details after update
        logger.log(Logger::LogLevel::DEBUG, "  Component %zu (after update): Mean[0]=%f, Cov[0,0]=%f",
                   i, comp.mean(0), comp.covariance(0,0));


        double predicted_vol_sq_for_likelihood = std::exp(comp.mean(0));
        double predicted_return_variance_for_likelihood = predicted_vol_sq_for_likelihood;


        if (predicted_return_variance_for_likelihood < small_positive_val) {
            predicted_return_variance_for_likelihood = small_positive_val;
        }


        double likelihood = (1.0 / std::sqrt(2.0 * M_PI * predicted_return_variance_for_likelihood)) *
                            std::exp(- (observed_return * observed_return) / (2.0 * predicted_return_variance_for_likelihood));

        current_likelihoods.push_back(likelihood);
        sum_weighted_likelihoods += comp.weight * likelihood;
    }

    // Normalize weights
    if (sum_weighted_likelihoods > 0) {
        for (size_t i = 0; i < components.size(); ++i) {
            components[i].weight = (components[i].weight * current_likelihoods[i]) / sum_weighted_likelihoods;
        }
    } else {
        logger.log(Logger::LogLevel::ERROR, "Sum of weighted likelihoods is zero after update. Filter divergence or extreme observation. Re-initializing filter state.");
        // Re-initialize with a single default component to prevent crash
        components.clear();
        Eigen::VectorXd initial_mean_reinit(1);
        double initial_daily_variance_reinit = filter_params.InitialFilterVolSq / model_params.DaysPerYear;
        if (initial_daily_variance_reinit <= 0) {
            initial_daily_variance_reinit = small_positive_val;
        }
        initial_mean_reinit << std::log(initial_daily_variance_reinit);
        Eigen::MatrixXd initial_covariance_reinit(1, 1);
        initial_covariance_reinit << filter_params.InitialFilterCov;
        components.emplace_back(initial_mean_reinit, initial_covariance_reinit, 1.0);
        return; // Skip pruning/merging/splitting for this step if re-initialized
    }

    // Post-update processing: prune, merge, and split components
    pruneComponents();
    mergeComponents();
    splitComponents(); // Call split after prune/merge
}

void AdaptiveGMF::pruneComponents() {
    // Remove components with weights below a threshold, then re-normalize.
    components.erase(
        std::remove_if(components.begin(), components.end(),
                       [&](const GaussianComponent& c) {
                           return c.weight < filter_params.MinComponentWeight;
                       }),
        components.end());

    double current_total_weight = 0.0;
    for (const auto& comp : components) {
        current_total_weight += comp.weight;
    }

    if (current_total_weight > 0) {
        for (auto& comp : components) {
            comp.weight /= current_total_weight;
        }
    } else {
        logger.log(Logger::LogLevel::ERROR, "All components pruned or sum of weights became zero. Re-initializing filter state.");
        components.clear();
        Eigen::VectorXd initial_mean_reinit(1);
        double initial_daily_variance_reinit = filter_params.InitialFilterVolSq / model_params.DaysPerYear;
        if (initial_daily_variance_reinit <= 0) { initial_daily_variance_reinit = small_positive_val; }
        initial_mean_reinit << std::log(initial_daily_variance_reinit);
        Eigen::MatrixXd initial_covariance_reinit(1, 1);
        initial_covariance_reinit << filter_params.InitialFilterCov;
        components.emplace_back(initial_mean_reinit, initial_covariance_reinit, 1.0);
    }


    if (components.size() > filter_params.MaxFilterComponents) {
        std::sort(components.begin(), components.end(),
                  [](const GaussianComponent& a, const GaussianComponent& b) {
                      return a.weight > b.weight; // Sort by weight descending
                  });

        // Create a new vector with only the top components
        std::vector<GaussianComponent> top_components;
        top_components.reserve(filter_params.MaxFilterComponents); // Pre-allocate memory

        for (size_t i = 0; i < filter_params.MaxFilterComponents; ++i) {
            top_components.push_back(components[i]);
        }
        components = top_components; // Assign the new vector back
        // --- END FIX ---

        // Re-normalize weights after resizing/reconstructing
        current_total_weight = 0.0;
        for (const auto& comp : components) {
            current_total_weight += comp.weight;
        }
        if (current_total_weight > 0.0) {
            for (auto& comp : components) {
                comp.weight /= current_total_weight;
            }
        }
    }
}


// Advanced merging strategy based on KL-divergence (or a similar distance metric)
void AdaptiveGMF::mergeComponents() {
    if (components.size() <= 1) {
        return; // Nothing to merge if 0 or 1 components
    }

    bool merged_in_iteration = true;
    while (merged_in_iteration && components.size() > 1) {
        merged_in_iteration = false;
        double min_merge_cost = std::numeric_limits<double>::max();
        int best_i = -1, best_j = -1;

        for (size_t i = 0; i < components.size(); ++i) {
            for (size_t j = i + 1; j < components.size(); ++j) {
                const GaussianComponent& comp1 = components[i];
                const GaussianComponent& comp2 = components[j];

                double merged_weight = comp1.weight + comp2.weight;
                Eigen::VectorXd merged_mean = (comp1.weight * comp1.mean + comp2.weight * comp2.mean) / merged_weight;

                Eigen::VectorXd diff_mean1 = comp1.mean - merged_mean;
                Eigen::VectorXd diff_mean2 = comp2.mean - merged_mean;
                Eigen::MatrixXd merged_covariance =
                    (comp1.weight * (comp1.covariance + diff_mean1 * diff_mean1.transpose()) +
                     comp2.weight * (comp2.covariance + diff_mean2 * diff_mean2.transpose())) / merged_weight;

                // Ensure merged_covariance is positive definite and not too small
                if (merged_covariance.determinant() <= 0 || merged_covariance(0,0) < small_positive_val) {
                    merged_covariance(0,0) = std::max(merged_covariance(0,0), small_positive_val);
                    if (merged_covariance.determinant() <= 0) {
                         merged_covariance(0,0) += std::numeric_limits<double>::epsilon();
                    }
                }

                double cost_ij = GaussianComponent::calculateKLDivergence(comp1, GaussianComponent(merged_mean, merged_covariance, 0.0)) * comp1.weight +
                                 GaussianComponent::calculateKLDivergence(comp2, GaussianComponent(merged_mean, merged_covariance, 0.0)) * comp2.weight;

                if (cost_ij < min_merge_cost) {
                    min_merge_cost = cost_ij;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i != -1 && min_merge_cost < filter_params.MergeThreshold) {
            GaussianComponent& comp1 = components[best_i];
            GaussianComponent& comp2 = components[best_j];

            double new_weight = comp1.weight + comp2.weight;
            Eigen::VectorXd new_mean = (comp1.weight * comp1.mean + comp2.weight * comp2.mean) / new_weight;

            Eigen::VectorXd diff_mean1 = comp1.mean - new_mean;
            Eigen::VectorXd diff_mean2 = comp2.mean - new_mean;
            Eigen::MatrixXd new_covariance = (comp1.weight * (comp1.covariance + diff_mean1 * diff_mean1.transpose()) +
                                            comp2.weight * (comp2.covariance + diff_mean2 * diff_mean2.transpose())) / new_weight;

            // Ensure new_covariance is positive definite and not too small after merge
            if (new_covariance.determinant() <= 0 || new_covariance(0,0) < small_positive_val) {
                new_covariance(0,0) = std::max(new_covariance(0,0), small_positive_val);
                if (new_covariance.determinant() <= 0) {
                    new_covariance(0,0) += std::numeric_limits<double>::epsilon();
                }
            }

            components[best_i] = GaussianComponent(new_mean, new_covariance, new_weight);
            components.erase(components.begin() + best_j);
            merged_in_iteration = true;
        } else {
            merged_in_iteration = false;
        }
    }
}


// Split step (for advanced GMF)
void AdaptiveGMF::splitComponents() {
    if (components.empty() || components.size() >= filter_params.MaxFilterComponents) {
        return; // Cannot split or already at max components
    }

    auto it_largest_weight = std::max_element(components.begin(), components.end(),
        [](const GaussianComponent& a, const GaussianComponent& b) {
            return a.weight < b.weight;
        });

    if (it_largest_weight != components.end()) {
        Eigen::VectorXd original_mean = it_largest_weight->mean;
        Eigen::MatrixXd original_covariance = it_largest_weight->covariance;
        double original_weight = it_largest_weight->weight;


        if (original_covariance(0,0) > 0.001 && components.size() < filter_params.MaxFilterComponents) {
            components.erase(it_largest_weight); // Erase original component FIRST

            // Create two new components by perturbing the mean
            // Perturbation scaled by sqrt(covariance) to be relative to spread
            double perturbation = std::sqrt(original_covariance(0,0)) * 0.5;
            // Ensure perturbation isn't too small to actually separate means
            perturbation = std::max(perturbation, small_positive_val * 100); // A bit larger than small_positive_val

            double new_weight = original_weight / 2.0;

            Eigen::VectorXd mean1 = original_mean;
            mean1(0) -= perturbation;
            Eigen::VectorXd mean2 = original_mean;
            mean2(0) += perturbation;

            Eigen::MatrixXd cov = original_covariance;
            // Ensure new component covariances are not too small
            cov(0,0) = std::max(cov(0,0), small_positive_val);

            components.emplace_back(mean1, cov, new_weight);
            components.emplace_back(mean2, cov, new_weight);

            logger.log(Logger::LogLevel::DEBUG, "Split a component. New total components: %zu", components.size());
        }
    }
}


// Accessor for current filter state (e.g., mean of the mixture)
double AdaptiveGMF::getEstimatedAnnualVolatility() const {
    if (components.empty()) {
        logger.log(Logger::LogLevel::WARNING, "No components to estimate annual volatility. Returning 0.0.");
        return 0.0;
    }

    Eigen::VectorXd mixture_mean(1);
    mixture_mean.setZero();
    double total_weight = 0.0;

    for (const auto& comp : components) {
        mixture_mean += comp.weight * comp.mean;
        total_weight += comp.weight;
    }

    if (total_weight > 0) {
        mixture_mean /= total_weight; // Normalize just in case weights aren't exactly 1 due to floating point
    } else {
        logger.log(Logger::LogLevel::WARNING, "Total weight is zero when estimating annual volatility. Returning 0.0.");
        return 0.0;
    }

    // Convert estimated log_v_t (mean of log_vol_sq) back to annualized volatility
    // Annual Vol = sqrt(exp(log_v_t_mean) * DaysPerYear)
    double estimated_daily_variance = std::exp(mixture_mean(0)); // This is estimated DAILY variance from log_v_t_mean

    // Ensure estimated daily variance isn't too small for square root
    estimated_daily_variance = std::max(estimated_daily_variance, small_positive_val);

    return std::sqrt(estimated_daily_variance * model_params.DaysPerYear); // Correct annualization
}

int AdaptiveGMF::getNumComponents() const {
    return components.size();
}
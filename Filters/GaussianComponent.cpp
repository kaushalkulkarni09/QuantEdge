// Filters/GaussianComponent.cpp
#include "Filters/GaussianComponent.h" // Self-include for declarations
#include <cmath> // For std::log, std::exp, std::sqrt
#include <limits> // For std::numeric_limits



// Constructor definition
GaussianComponent::GaussianComponent(const Eigen::VectorXd& m, const Eigen::MatrixXd& cov, double w)
    : mean(m), covariance(cov), weight(w) {

    if (covariance.determinant() <= 0) {
        covariance(0,0) += std::numeric_limits<double>::epsilon();
    }
}

// Static function definition for Kullback-Leibler (KL) Divergence between two 1D Gaussian distributions.
// KL(P || Q) for P = N(mu_p, sigma2_p) and Q = N(mu_q, sigma2_q) is:
// KL = 0.5 * ( (sigma2_p + (mu_p - mu_q)^2) / sigma2_q - 1 - log(sigma2_p / sigma2_q) )
double GaussianComponent::calculateKLDivergence(const GaussianComponent& p, const GaussianComponent& q) {
    // Extract means and variances from the 1D Eigen vectors/matrices
    double mu_p = p.mean(0);
    double sigma2_p = p.covariance(0,0); // Access the single element for 1D

    double mu_q = q.mean(0);
    double sigma2_q = q.covariance(0,0); // Access the single element for 1D


    if (sigma2_p < std::numeric_limits<double>::epsilon()) sigma2_p = std::numeric_limits<double>::epsilon();
    if (sigma2_q < std::numeric_limits<double>::epsilon()) sigma2_q = std::numeric_limits<double>::epsilon();

    double diff_mu = mu_p - mu_q;

    // Apply the KL-Divergence formula for 1D Gaussians
    return 0.5 * ( (sigma2_p + diff_mu * diff_mu) / sigma2_q - 1.0 - std::log(sigma2_p / sigma2_q) );
}

// Probability Density Function (PDF) calculation for a 1D Gaussian component.

double GaussianComponent::pdf(double x) const {
    double mu = mean(0);
    double sigma_sq = covariance(0,0);


    if (sigma_sq < std::numeric_limits<double>::epsilon()) {
        return (x == mu) ? std::numeric_limits<double>::max() : 0.0;
    }

    // Standard PDF formula for a 1D Gaussian distribution
    return (1.0 / std::sqrt(2.0 * M_PI * sigma_sq)) * std::exp(-0.5 * (x - mu) * (x - mu) / sigma_sq);
}

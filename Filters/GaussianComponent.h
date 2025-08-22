// Filters/GaussianComponent.h
#pragma once

#include <Eigen/Dense> // For Eigen::VectorXd and Eigen::MatrixXd
// No custom includes needed here for now.

// Define M_PI if not already defined (common in math libraries, used in PDF)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class GaussianComponent {
public:
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
    double weight;

    // Constructor declaration
    GaussianComponent(const Eigen::VectorXd& m, const Eigen::MatrixXd& cov, double w);

    // Static function declaration for KL-Divergence.
    // The implementation (definition) will be in GaussianComponent.cpp
    static double calculateKLDivergence(const GaussianComponent& p, const GaussianComponent& q);

    // Member function declaration for Probability Density Function (PDF).
    // The implementation (definition) will be in GaussianComponent.cpp
    double pdf(double x) const;
};

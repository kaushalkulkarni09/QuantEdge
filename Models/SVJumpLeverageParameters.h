// Models/SVJumpLeverageParameters.h
#pragma once

// This file defines the parameters for the Stochastic Volatility with Jumps and Leverage (SVJL) model.


struct ModelParameters {
    double Kappa;     // Mean reversion rate for log-variance (x_t = log(v_t^2))
    double Theta;     // Long-run mean of log-variance (x_t = log(v_t^2))
    double SigmaV;    // Volatility of volatility (vol of vol) for log-variance process
    double Rho;       // Correlation between asset returns and volatility (leverage effect)
    double Lambda;    // Jump intensity (average number of jumps per unit time)
    double MuJ;       // Mean of jump size (log return)
    double SigmaJ;    // Standard deviation of jump size (log return)
    double RiskFreeRate; // Risk-free interest rate
    int DaysPerYear; // Number of trading days per year (for annualization)
};
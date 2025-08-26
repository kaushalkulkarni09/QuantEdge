# **Adaptive Volatility Modeling with Dynamic Filter**

This project implements an adaptive filter to estimate the stochastic volatility of financial assets. Unlike traditional models, this filter doesn't treat volatility as a single, fixed value but as a dynamic variable that evolves over time. It operates by breaking down volatility into a series of weighted scenarios, each representing a possible market state.
The model's key strength is its ability to learn and adapt in real time. It continuously refines its understanding of the market by adding new scenarios when a new trend emerges and merging or removing scenarios when they're no longer relevant. This process ensures the model remains both accurate and computationally efficient, even as market conditions shift.
The project utilizes a hybrid C++/Python pipeline: the high-performance C++ core handles the filter's intensive calculations, while Python is used for data fetching and preprocessing, demonstrating a robust and efficient architecture.


**Core Components**

The project is structured into several key C++ and Python files that work together to form the application. Below the most important files from a execution perspective will be discussed:

* **Filters/AdaptiveGMF.cpp:** It contains the core C++ logic for the adaptive filter.

The filter's engine works in a two-step cycle: it first predicts what the market's volatility will be, then it updates that prediction using the latest daily data. This prediction is based on the idea that volatility tends to swing back toward a long-term average, rather than just moving randomly.

The model's real intelligence lies in its ability to manage its own complexity to stay efficient and accurate. It constantly monitors the different scenarios it's considering for volatility and takes three key actions:

  * **Pruning:** It automatically removes any scenarios that are no longer relevant to the market's behavior.
  * **Merging:** It combines scenarios that are too similar to avoid unnecessary calculations and keep the model concise.
  * **Splitting:** When the market exhibits a new trend, it creates new scenarios to make sure it captures every possibility.

**download_data.py:**
  * This script uses the popular yfinance library to download historical stock data.
  * It calculates daily log returns from the adjusted closing prices.
  * The processed log returns are saved to a CSV file
  * The script is designed to be invoked directly from the C++ code, showcasing integration between C++ and Python.

**CMakeLists.txt:**
  * It uses cmake to manage the build process.
  * It correctly configures the project to find and link against the Eigen library (for linear algebra) and the Python 3.9 framework (for data fetching).
  * The configuration for macOS (if(APPLE)) ensures that the rpath (runpath) is set correctly, allowing the executable to find the necessary Python dynamic libraries and modules at runtime.
  



**Data Analysis and Key Findings**

The analysis spans over 1,200 days of data for Apple (AAPL), revealing a clear and consistent pattern of model performance. The following table provides a chronological overview of key data points, illustrating the model's behavior across different market phases.

| Day | Log Returns | Volatility (G) | Vol # components |
| :--- | :--- | :--- | :--- |
| $1$ | $0.00049449$ | $0.22234105$ | $0$ |
| $50$ | $0.02271501$ | $0.24871910$ | $0$ |
| $100$ | $-0.0007702$ | $0.2239510$ | $1$ |
| $150$ | $0.00256263$ | $0.2387211$ | $2$ |
| $200$ | $0.00215856$ | $0.2401611$ | $2$ |
| $250$ | $0.00000000$ | $0.2376910$ | $3$ |
| $300$ | $0.00694415$ | $0.2981810$ | $3$ |
| $350$ | $-0.0121015$ | $0.228609$ | $4$ |
| $400$ | $-0.0163877$ | $0.2381510$ | $4$ |
| $450$ | $0.02002369$ | $0.236449$ | $5$ |
| $500$ | $0.02271501$ | $0.248719$ | $5$ |
| $550$ | $0.03439908$ | $0.263479$ | $6$ |
| $600$ | $-0.0249502$ | $0.308468$ | $6$ |
| $650$ | $-0.0093305$ | $0.313739$ | $7$ |
| $700$ | $-0.0046155$ | $0.3184110$ | $7$ |
| $750$ | $-0.0240596$ | $0.3503910$ | $8$ |
| $800$ | $0.00834256$ | $0.3214511$ | $8$ |
| $850$ | $0.01357364$ | $0.2859611$ | $9$ |
| $900$ | $-0.0042850$ | $0.2512399$ | $9$ |
| $950$ | $0.00738498$ | $0.2277210$ | $10$ |
| $1000$ | $-0.0007702$ | $0.2239510$ | $10$ |
| $1050$ | $-0.0058955$ | $0.2160610$ | $11$ |
| $1100$ | $0.00063224$ | $0.2285191$ | $11$ |
| $1150$ | $0.00256263$ | $0.2387211$ | $12$ |
| $1200$ | $0.01656051$ | $0.2376511$ | $2$ |
| $1256$ | $-0.01335222$ | $0.2145610$ | $1$ |


**The Full Narrative of Model Performance**

**Phase 1 (Days 1-500):** The model begins with moderate volatility around 22% and a low number of components. The component count, which reflects the complexity of the GMM, gradually increases as the model learns to represent the volatility distribution more accurately. This phase demonstrates the filter's initial learning and adaptation as it builds a more sophisticated model of market behavior.

**Phase 2 (Days 500-750):** This period shows a significant spike in market volatility, with the volatility value steadily climbing to a peak of over 0.35 on Day 750. In this high-stress period, the model manages the GMM's complexity, showing that it can maintain its stability even with a growing number of components. 

**Phase 3 (Days 800-1100):** Following the volatility peak, the model's effectiveness becomes apparent. The volatility number begins a clear and consistent descent, stabilizing near the 22% mark. As the market calms, the filter efficiently manages its components, pruning and merging them to maintain an optimal representation of the volatility.

**Phase 4 (Days 1150-1256):** In the final 100 days, the model's performance culminates. Volatility remains consistently low, and the filter demonstrates its ability to dynamically prune components to reflect a simplified market state. The component count drastically drops from 12 to 1, and the final volatility on Day 1256 settles at a stable 0.21456, showing the model has converged on a highly reliable and efficient state.

**Dependencies:**

This project relies on the following external libraries:

**C++ Libraries:**

 * **Eigen 3.4.0:** A C++ library for linear algebra.
 * **C++ Standard Library:** The code leverages standard libraries like < cmath >, < vector >, and < numeric >.

**Python Libraries:**

* **yfinance:** For downloading historical market data.
* **pandas:** For data manipulation and log return calculation.
* **numpy:** For numerical operations.
 


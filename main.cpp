// main.cpp
// main.cpp
#include <iostream>
#include <vector>
#include <string>
#include <random> // For C++11 random number generation
#include <iomanip> // For std::fixed, std::setprecision, std::setw, std::right
#include <sstream> // For std::stringstream
#include <filesystem> // For std::filesystem::current_path

// Custom headers (paths relative to project root)
#include "Config/ConfigParser.h"
#include "Utils/Logger.h"
#include "Data/DataSimulator.h"
#include "Data/DataReader.h"
#include "IO/OutputWriter.h"
#include "Filters/AdaptiveGMF.h"
#include "IO/PythonDataFetcher.h" // Includes Python.h
#include "Models/SVJumpLeverageParameters.h"
#include "Utils/MathUtils.h" // Include MathUtils

int main(int argc, char* argv[]) {
    // 1. Initialize Logger
    Logger logger("application.log", Logger::INFO);
    logger.log(Logger::INFO, "Application startup. Initializing logger.");
    logger.log(Logger::INFO, "-----------------------------------------------------------------------");

    // 2. Parse Configuration
    ConfigParser config("config.ini", logger);
    if (!config.parse()) {
        logger.log(Logger::ERROR, "Failed to parse config file: config.ini. Exiting.");
        return 1;
    }
    logger.log(Logger::INFO, "Successfully parsed config file: config.ini");

    // Re-initialize logger with config settings
    logger.reinitialize(config.getLoggingFilename(), config.getLoggingLevelThreshold());
    logger.log(Logger::INFO, "Logger re-initialized with config settings. Log file: %s, Level: %s",
               config.getLoggingFilename().c_str(), Logger::levelToString(config.getLoggingLevelThreshold()).c_str());

    // Get all parameters from config
    DataSourceParameters ds_params = config.getDataSourceParameters();
    SimulationParameters sim_params = config.getSimulationParameters();
    ModelParameters model_params = config.getModelParameters();
    FilterParameters filter_params = config.getFilterParameters();
    OutputParameters output_params = config.getOutputParameters();

    logger.log(Logger::INFO, "DataSource configured: SimulatedData=%s, UsePythonFetcher=%s, InputFile=%s, ReturnCol=%d",
               ds_params.UseSimulatedData ? "true" : "false",
               ds_params.UsePythonFetcher ? "true" : "false",
               ds_params.InputFilename.c_str(), ds_params.ReturnColumnIndex);
    logger.log(Logger::INFO, "Simulation configured: Years=%d, DaysPerYear=%d, InitialTrueVolSq=%f",
               sim_params.NumYears, sim_params.NumTradingDaysPerYear, sim_params.InitialTrueVolSq);
    logger.log(Logger::INFO, "Model Parameters loaded from config.");
    logger.log(Logger::INFO, "Filter Parameters loaded from config.");

    // 3. Data Acquisition
    std::vector<double> observed_returns;
    std::vector<double> true_annual_volatilities;

    // PythonDataFetcher instance for fetching initial data
    PythonDataFetcher data_fetcher_instance(config.getPythonScriptPath(),
                                            config.getPythonFunctionName(),
                                            config.getTempOutputCSVPath());

    // Flag to track if Python was initialized by this main function, for proper finalization
    bool python_initialized_locally = false;

    if (ds_params.UseSimulatedData) {
        logger.log(Logger::INFO, "Using Simulated Data (from config).");
        DataSimulator simulator(sim_params, model_params, logger);
        observed_returns = simulator.generateData();
        true_annual_volatilities = simulator.getTrueVolatilities();

        // If using simulated data, Python is NOT initialized by PythonDataFetcher.
        // Initialize Python interpreter here for plotting.
        logger.log(Logger::INFO, "Python interpreter not initialized yet, initializing for plotting (simulated data case)...");
        Py_Initialize();
        if (!Py_IsInitialized()) {
            logger.log(Logger::ERROR, "Failed to initialize Python interpreter for plotting. Exiting plot attempt.");
            PyErr_Print();
            return 1;
        }
        python_initialized_locally = true;

    } else if (ds_params.UsePythonFetcher) {
        logger.log(Logger::INFO, "Attempting to fetch data using PythonDataFetcher...");

        logger.log(Logger::INFO, "Python Script: %s", config.getPythonScriptPath().c_str());
        logger.log(Logger::INFO, "Python Function: %s", config.getPythonFunctionName().c_str());
        logger.log(Logger::INFO, "Ticker: %s, Dates: %s to %s",
                   config.getTicker().c_str(), config.getStartDate().c_str(), config.getEndDate().c_str());
        logger.log(Logger::INFO, "Temporary Output CSV: %s", config.getTempOutputCSVPath().c_str());

        try {
            if (!data_fetcher_instance.initialize()) {
                logger.log(Logger::ERROR, "Failed to initialize PythonDataFetcher. Exiting.");
                return 1;
            }

            if (!data_fetcher_instance.fetchData(config.getTicker(), config.getStartDate(), config.getEndDate())) {
                logger.log(Logger::ERROR, "Failed to fetch data using Python script. Exiting.");
                return 1;
            }
            logger.log(Logger::INFO, "Python data fetching successful. Loading data from temporary CSV: %s", config.getTempOutputCSVPath().c_str());

            DataReader data_reader(config.getTempOutputCSVPath(), ds_params.ReturnColumnIndex, logger);
            observed_returns = data_reader.readData();
            logger.log(Logger::INFO, "Successfully loaded %zu data points from Python-fetched CSV.", observed_returns.size());

        } catch (const std::exception& e) {
            logger.log(Logger::ERROR, "An error occurred during Python data fetching: %s", e.what());
            return 1;
        }
    } else {
        logger.log(Logger::INFO, "Loading data from InputFile: %s (from config).", ds_params.InputFilename.c_str());
        DataReader data_reader(ds_params.InputFilename, ds_params.ReturnColumnIndex, logger);
        observed_returns = data_reader.readData();
        logger.log(Logger::INFO, "Successfully loaded %zu data points from %s.", observed_returns.size(), ds_params.InputFilename.c_str());

        // If using local CSV, Python is NOT initialized.
        logger.log(Logger::INFO, "Python interpreter not initialized yet, initializing for plotting (local CSV case)...");
        Py_Initialize();
        if (!Py_IsInitialized()) {
            logger.log(Logger::ERROR, "Failed to initialize Python interpreter for plotting. Exiting plot attempt.");
            PyErr_Print();
            return 1;
        }
        python_initialized_locally = true;
    }

    if (observed_returns.empty()) {
        logger.log(Logger::ERROR, "No observed returns available. Exiting.");
        return 1;
    }
    logger.log(Logger::INFO, "Processing %zu data points.", observed_returns.size());

    // 4. Initialize and Run Filter
    AdaptiveGMF filter(filter_params, model_params, logger);
    logger.log(Logger::INFO, "Filter initialized with %d components.", filter.getNumComponents());
    logger.log(Logger::INFO, "Max components allowed: %d", filter_params.MaxFilterComponents);
    logger.log(Logger::INFO, "Merge Threshold (KL-like): %f", filter_params.MergeThreshold);
    logger.log(Logger::INFO, "Min Component Weight: %f", filter_params.MinComponentWeight);

    OutputWriter writer(output_params.OutputFilename, logger);
    logger.log(Logger::INFO, "Output writer initialized for file: %s (from config).", output_params.OutputFilename.c_str());

    writer.writeLine("Day\tObserved Return\tTrue Ann. Vol (Simulated Only)\tFilter Ann. Vol\t# Components");
    writer.writeLine("---------------------------------------------------------------------------------------------------");

    logger.log(Logger::INFO, "\n--- Filtering Results ---");
    for (size_t day = 0; day < observed_returns.size(); ++day) {
        filter.predict();
        filter.update(observed_returns[day]);

        double filter_ann_vol = filter.getEstimatedAnnualVolatility();
        int num_components = filter.getNumComponents();

        // --- Start of Formatted Output Block for better alignment ---
        std::stringstream ss;
        // Set fixed-point notation for consistent decimal places
        ss << std::fixed;

        // Day column: Right-aligned, fixed width (e.g., 5 characters), followed by a tab
        ss << std::right << std::setw(5) << day + 1 << "\t";

        // Observed Return column: Right-aligned, fixed width (e.g., 15 characters), 9 decimal places
        ss << std::right << std::setw(15) << std::setprecision(9) << observed_returns[day] << "\t";

        // True Annual Volatility (Simulated Only) column: Right-aligned, width (e.g., 20 characters), 9 decimal places or "N/A"
        if (ds_params.UseSimulatedData && day < true_annual_volatilities.size()) {
            ss << std::right << std::setw(20) << std::setprecision(9) << true_annual_volatilities[day] << "\t";
        } else {
            // Fill with spaces if "N/A" to match desired width for proper alignment
            ss << std::right << std::setw(20) << "N/A" << "\t";
        }

        // Filter Annual Volatility column: Right-aligned, width (e.g., 12 characters), 5 decimal places
        ss << std::right << std::setw(12) << std::setprecision(5) << filter_ann_vol << "\t";

        // # Components column: Right-aligned, width (e.g., 5 characters)
        ss << std::right << std::setw(5) << num_components;
        // --- End of Formatted Output Block ---

        writer.writeLine(ss.str());

        // Log to console less frequently, or log just a few lines for brevity
        // This condition prints the first 10, then every 1/10th of the data, and the last line.
        if ((day + 1) % (observed_returns.size() / 10 + 1) == 0 || day < 10 || day == observed_returns.size() - 1) {
            logger.log(Logger::INFO, "%s", ss.str().c_str());
        }
    }

    logger.log(Logger::INFO, "Filtering complete. Results saved to %s", output_params.OutputFilename.c_str());

    // --- NEW: Call Python plotting function from C++ ---
    logger.log(Logger::INFO, "Attempting to generate plots using Python...");

    // Determine the path to the output CSV file for Python plotting
    // When running from CLion's debug build, std::filesystem::current_path() is usually the 'cmake-build-debug' directory.
    // The filter_results.csv is also in this directory.
    std::filesystem::path plot_csv_full_path = std::filesystem::current_path() / output_params.OutputFilename;

    // Get the project root path. This is necessary because plot_results.py is in the project root.
    // If current_path() is cmake-build-debug, then parent_path() is the project root.
    std::filesystem::path project_root_path = std::filesystem::current_path().parent_path();
    std::string plot_module_name = "plot_results"; // Module name is filename without .py

    // --- CRITICAL PATH ADDITION FOR PLOTTING MODULE ---
    // Ensure project root is in Python's sys.path so it can find plot_results.py
    if (Py_IsInitialized()) { // Only proceed if Python interpreter is active
        PyObject* sysPath = PySys_GetObject("path");
        if (sysPath) {
            PyObject* py_project_root_path = PyUnicode_DecodeFSDefault(project_root_path.string().c_str());
            if (py_project_root_path) {
                if (PyList_Append(sysPath, py_project_root_path) < 0) {
                    logger.log(Logger::WARNING, "Failed to add project root to Python sys.path for plotting.");
                    PyErr_Print();
                } else {
                    logger.log(Logger::INFO, "Added project root to Python sys.path: %s", project_root_path.string().c_str());
                }
                Py_XDECREF(py_project_root_path);
            } else {
                logger.log(Logger::WARNING, "Failed to convert project root path to Python string.");
                PyErr_Print();
            }

            // Also ensure the virtual environment's site-packages are on path, especially if python_initialized_locally was true
            std::filesystem::path venv_site_packages_path = project_root_path / ".venv" / "lib" / "python3.9" / "site-packages";
            if (std::filesystem::exists(venv_site_packages_path)) {
                PyObject* py_venv_site_packages_path = PyUnicode_DecodeFSDefault(venv_site_packages_path.string().c_str());
                if (py_venv_site_packages_path) {
                    if (PyList_Append(sysPath, py_venv_site_packages_path) < 0) {
                        logger.log(Logger::WARNING, "Failed to add venv site-packages to Python sys.path for plotting.");
                        PyErr_Print();
                    } else {
                        logger.log(Logger::INFO, "Added venv site-packages to Python sys.path: %s", venv_site_packages_path.string().c_str());
                    }
                    Py_XDECREF(py_venv_site_packages_path);
                }
            } else {
                logger.log(Logger::WARNING, "Virtual environment site-packages not found at %s. Plotting might fail if matplotlib/pandas missing.", venv_site_packages_path.string().c_str());
            }
        } else {
            logger.log(Logger::WARNING, "Could not get Python sys.path for adding directories.");
            PyErr_Print();
        }
    }


    // Now, import the plotting module and call its function
    PyObject* pPlotModule = PyImport_ImportModule(plot_module_name.c_str());
    if (!pPlotModule) {
        logger.log(Logger::ERROR, "Failed to load Python plotting module: %s", plot_module_name.c_str());
        // Use printError from data_fetcher_instance if it was initialized by it, else PyErr_Print
        if (data_fetcher_instance.isInitialized()) {
            data_fetcher_instance.printError();
        } else {
            PyErr_Print();
        }
    } else {
        PyObject* pPlotFunc = PyObject_GetAttrString(pPlotModule, "generate_plots");
        if (!pPlotFunc || !PyCallable_Check(pPlotFunc)) {
            logger.log(Logger::ERROR, "Cannot find 'generate_plots' function in 'plot_results.py' or it is not callable.");
            if (data_fetcher_instance.isInitialized()) {
                data_fetcher_instance.printError();
            } else if (PyErr_Occurred()) {
                PyErr_Print();
            }
            Py_XDECREF(pPlotModule);
        } else {
            PyObject* pArgs = PyTuple_New(1);
            PyObject* py_csv_path = PyUnicode_FromString(plot_csv_full_path.string().c_str());
            if (py_csv_path) {
                PyTuple_SetItem(pArgs, 0, py_csv_path); // PyTuple_SetItem steals reference
                PyObject* pPlotResult = PyObject_CallObject(pPlotFunc, pArgs);
                Py_DECREF(pArgs); // Done with tuple

                if (pPlotResult == nullptr) {
                    logger.log(Logger::ERROR, "Call to Python plotting function 'generate_plots' failed.");
                    if (data_fetcher_instance.isInitialized()) {
                        data_fetcher_instance.printError();
                    } else {
                        PyErr_Print();
                    }
                } else if (pPlotResult == Py_False) {
                    logger.log(Logger::WARNING, "Python plotting function reported failure (returned False).");
                } else {
                    logger.log(Logger::INFO, "Python plotting function called successfully. Plots should appear.");
                }
                Py_XDECREF(pPlotResult);
            } else {
                logger.log(Logger::ERROR, "Failed to convert CSV path to Python string.");
                Py_XDECREF(pArgs);
            }
            Py_XDECREF(pPlotFunc);
        }
        Py_XDECREF(pPlotModule);
    }

    // Finalize Python interpreter ONLY if it was initialized locally in this main.cpp.
    // If PythonDataFetcher initialized it, its destructor will handle finalization.
    if (python_initialized_locally) {
        Py_Finalize();
        logger.log(Logger::INFO, "Python interpreter finalized locally.");
    }

    return 0;
}

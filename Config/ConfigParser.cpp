// Config/ConfigParser.cpp
#include "Config/ConfigParser.h" // Self-include
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cmath> // IMPORTANT: Added for std::log

// Helper to trim whitespace from a string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(first, (last - first + 1));
}

ConfigParser::ConfigParser(const std::string& filename, Logger& logger)
    : config_filename(filename), logger(logger) {
    // Initialize with default values

    // Define default trading days per year for consistent calculations
    // This value will be used for default initializations and passed to model_params.
    int default_days_per_year = 252; // Standard for equities

    // --- DataSource Parameters Defaults ---
    ds_params = {false, false, "input_data.csv", 0};

    // --- Simulation Parameters Defaults ---
    // InitialTrueVolSq is assumed to be an ANNUAL variance (e.g., 0.04 for 20% annual vol)
    sim_params = {
        1,                         // NumYears
        default_days_per_year,     // NumTradingDaysPerYear
        0.04                       // InitialTrueVolSq (Annual Variance, e.g., 20% annual volatility squared)
    };

    // --- Model Parameters Defaults ---
    // Calculate Theta based on a sensible target *annual* volatility.
    // Theta is the long-run mean of log(daily variance).

    double target_annual_vol_for_theta = 0.20;
    double target_annual_variance_for_theta = target_annual_vol_for_theta * target_annual_vol_for_theta; // 0.04
    double target_daily_variance_for_theta = target_annual_variance_for_theta / default_days_per_year;
    double calculated_theta = std::log(target_daily_variance_for_theta); // This will be approx -8.7496

    model_params = {
        2.0,                  // Kappa: Mean reversion rate for log-variance
        calculated_theta,     // Theta: Long-run mean of log(daily variance)
        0.2,                  // SigmaV: Volatility of volatility for log-variance process
        -0.7,                 // Rho: Correlation between asset returns and volatility (leverage effect)
        0.1,                  // Lambda: Jump intensity (average number of jumps per unit time)
        -0.02,                // MuJ: Mean of jump size (log return)
        0.05,                 // SigmaJ: Standard deviation of jump size (log return)
        0.02,                 // RiskFreeRate: Risk-free interest rate
        default_days_per_year // DaysPerYear: Number of trading days per year
    };

    // --- Filter Parameters Defaults ---
    // InitialFilterVolSq is assumed to be an ANNUAL variance.
    // This value (0.032) corresponds to sqrt(0.032) = ~0.1789 or ~17.9% annual volatility.
    filter_params = {
        0.032, // InitialFilterVolSq (Annual Variance)
        0.01,  // InitialFilterCov
        1,     // NumInitialComponents
        50,    // MaxFilterComponents
        0.005, // MergeThreshold
        0.00000001 // MinComponentWeight
    };

    // --- Output Parameters Defaults ---
    output_params = {"filter_results.csv"};

    // --- PythonFetcher Parameters Defaults ---
    python_script_path = "download_data.py";
    python_function_name = "download_and_save_data";
    ticker = "SPY"; // Default ticker
    start_date = "2020-01-01";
    end_date = "2024-12-31";
    temp_output_csv_path = "fetched_data.csv";

    // --- Logging Parameters Defaults ---
    logging_filename = "application.log";
    logging_level_threshold = Logger::INFO;
}

bool ConfigParser::parse() {
    std::ifstream file(config_filename);
    if (!file.is_open()) {
        logger.log(Logger::ERROR, "Could not open config file: %s", config_filename.c_str());
        return false;
    }

    std::string line;
    std::string current_section;
    int line_num = 0;

    while (std::getline(file, line)) {
        line_num++;
        line = trim(line);

        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue; // Skip empty lines and comments
        }

        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.length() - 2);
            config_map[current_section] = std::map<std::string, std::string>();
        } else {
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = trim(line.substr(0, eq_pos));
                std::string value = trim(line.substr(eq_pos + 1));
                if (!current_section.empty()) {
                    config_map[current_section][key] = value;
                } else {
                    logger.log(Logger::WARNING, "Orphan key-value pair '%s' at line %d in config file. No section defined.", key.c_str(), line_num);
                }
            } else {
                logger.log(Logger::WARNING, "Invalid line format '%s' at line %d in config file. Skipping.", line.c_str(), line_num);
            }
        }
    }
    file.close();

    // Populate parameter structs from the parsed map (overriding defaults if found in config file)
    ds_params.UseSimulatedData = getBool("DataSource", "UseSimulatedData", ds_params.UseSimulatedData);
    ds_params.UsePythonFetcher = getBool("DataSource", "UsePythonFetcher", ds_params.UsePythonFetcher);
    ds_params.InputFilename = getString("DataSource", "InputFilename", ds_params.InputFilename);
    ds_params.ReturnColumnIndex = getInt("DataSource", "ReturnColumnIndex", ds_params.ReturnColumnIndex);

    sim_params.NumYears = getInt("Simulation", "NumYears", sim_params.NumYears);
    sim_params.NumTradingDaysPerYear = getInt("Simulation", "NumTradingDaysPerYear", sim_params.NumTradingDaysPerYear);
    sim_params.InitialTrueVolSq = getDouble("Simulation", "InitialTrueVolSq", sim_params.InitialTrueVolSq);

    model_params.Kappa = getDouble("ModelParameters", "Kappa", model_params.Kappa);
    model_params.Theta = getDouble("ModelParameters", "Theta", model_params.Theta);
    model_params.SigmaV = getDouble("ModelParameters", "SigmaV", model_params.SigmaV);
    model_params.Rho = getDouble("ModelParameters", "Rho", model_params.Rho);
    model_params.Lambda = getDouble("ModelParameters", "Lambda", model_params.Lambda);
    model_params.MuJ = getDouble("ModelParameters", "MuJ", model_params.MuJ);
    model_params.SigmaJ = getDouble("ModelParameters", "SigmaJ", model_params.SigmaJ);
    model_params.RiskFreeRate = getDouble("ModelParameters", "RiskFreeRate", model_params.RiskFreeRate);
    // model_params.DaysPerYear should always align with NumTradingDaysPerYear from Simulation
    model_params.DaysPerYear = sim_params.NumTradingDaysPerYear;

    filter_params.InitialFilterVolSq = getDouble("FilterParameters", "InitialFilterVolSq", filter_params.InitialFilterVolSq);
    filter_params.InitialFilterCov = getDouble("FilterParameters", "InitialFilterCov", filter_params.InitialFilterCov);
    filter_params.NumInitialComponents = getInt("FilterParameters", "NumInitialComponents", filter_params.NumInitialComponents);
    filter_params.MaxFilterComponents = getInt("FilterParameters", "MaxFilterComponents", filter_params.MaxFilterComponents);
    filter_params.MergeThreshold = getDouble("FilterParameters", "MergeThreshold", filter_params.MergeThreshold);
    filter_params.MinComponentWeight = getDouble("FilterParameters", "MinComponentWeight", filter_params.MinComponentWeight);

    output_params.OutputFilename = getString("Output", "OutputFilename", output_params.OutputFilename);

    python_script_path = getString("PythonFetcher", "PythonScriptPath", python_script_path);
    python_function_name = getString("PythonFetcher", "PythonFunctionName", python_function_name);
    ticker = getString("PythonFetcher", "Ticker", ticker);
    start_date = getString("PythonFetcher", "StartDate", start_date);
    end_date = getString("PythonFetcher", "EndDate", end_date);
    temp_output_csv_path = getString("PythonFetcher", "TempOutputCSV", temp_output_csv_path);

    logging_filename = getString("Logging", "LogFilename", logging_filename);
    parseLogLevel(getString("Logging", "LogLevelThreshold", Logger::levelToString(logging_level_threshold)));

    return true;
}

bool ConfigParser::getBool(const std::string& section, const std::string& key, bool default_val) {
    if (config_map.count(section) && config_map[section].count(key)) {
        std::string val = trim(config_map[section][key]);
        std::transform(val.begin(), val.end(), val.begin(), ::tolower);
        return (val == "true" || val == "1");
    }
    return default_val;
}

int ConfigParser::getInt(const std::string& section, const std::string& key, int default_val) {
    if (config_map.count(section) && config_map[section].count(key)) {
        try {
            return std::stoi(trim(config_map[section][key]));
        } catch (const std::invalid_argument& e) {
            logger.log(Logger::WARNING, "Invalid integer format for '%s' in section '%s'. Using default %d. Error: %s", key.c_str(), section.c_str(), default_val, e.what());
        } catch (const std::out_of_range& e) {
            logger.log(Logger::WARNING, "Integer out of range for '%s' in section '%s'. Using default %d. Error: %s", key.c_str(), section.c_str(), default_val, e.what());
        }
    }
    return default_val;
}

double ConfigParser::getDouble(const std::string& section, const std::string& key, double default_val) {
    if (config_map.count(section) && config_map[section].count(key)) {
        try {
            return std::stod(trim(config_map[section][key]));
        } catch (const std::invalid_argument& e) {
            logger.log(Logger::WARNING, "Invalid double format for '%s' in section '%s'. Using default %f. Error: %s", key.c_str(), section.c_str(), default_val, e.what());
        } catch (const std::out_of_range& e) {
            logger.log(Logger::WARNING, "Double out of range for '%s' in section '%s'. Using default %f. Error: %s", key.c_str(), section.c_str(), default_val, e.what());
        }
    }
    return default_val;
}

std::string ConfigParser::getString(const std::string& section, const std::string& key, const std::string& default_val) {
    if (config_map.count(section) && config_map[section].count(key)) {
        return trim(config_map[section][key]);
    }
    return default_val;
}

void ConfigParser::parseLogLevel(const std::string& level_str) {
    std::string lower_level_str = level_str;
    std::transform(lower_level_str.begin(), lower_level_str.end(), lower_level_str.begin(), ::tolower);

    if (lower_level_str == "debug") {
        logging_level_threshold = Logger::DEBUG;
    } else if (lower_level_str == "info") {
        logging_level_threshold = Logger::INFO;
    } else if (lower_level_str == "warning") {
        logging_level_threshold = Logger::WARNING;
    } else if (lower_level_str == "error") {
        logging_level_threshold = Logger::ERROR;
    } else {
        logger.log(Logger::WARNING, "Unknown log level '%s' in config. Using default INFO.", level_str.c_str());
        logging_level_threshold = Logger::INFO;
    }
}

DataSourceParameters ConfigParser::getDataSourceParameters() const { return ds_params; }
SimulationParameters ConfigParser::getSimulationParameters() const { return sim_params; }
ModelParameters ConfigParser::getModelParameters() const { return model_params; }
FilterParameters ConfigParser::getFilterParameters() const { return filter_params; }
OutputParameters ConfigParser::getOutputParameters() const { return output_params; }

std::string ConfigParser::getPythonScriptPath() const { return python_script_path; }
std::string ConfigParser::getPythonFunctionName() const { return python_function_name; }
std::string ConfigParser::getTicker() const { return ticker; }
std::string ConfigParser::getStartDate() const { return start_date; }
std::string ConfigParser::getEndDate() const { return end_date; }
std::string ConfigParser::getTempOutputCSVPath() const { return temp_output_csv_path; }

std::string ConfigParser::getLoggingFilename() const { return logging_filename; }
Logger::LogLevel ConfigParser::getLoggingLevelThreshold() const { return logging_level_threshold; }
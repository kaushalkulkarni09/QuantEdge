// Config/ConfigParser.h
#pragma once

#include <string>
#include <map>
#include <vector>
#include <stdexcept>

// Custom headers (paths relative to project root)
#include "Utils/Logger.h" // Logger class
#include "Models/SVJumpLeverageParameters.h" // ModelParameters struct is defined here

// Parameter structs for clear organization
struct DataSourceParameters {
    bool UseSimulatedData;
    bool UsePythonFetcher;
    std::string InputFilename;
    int ReturnColumnIndex;
};

struct SimulationParameters {
    int NumYears;
    int NumTradingDaysPerYear;
    double InitialTrueVolSq;
};

struct FilterParameters {
    double InitialFilterVolSq;
    double InitialFilterCov;
    int NumInitialComponents;
    int MaxFilterComponents;
    double MergeThreshold; // Advanced: KL-divergence like threshold
    double MinComponentWeight;
};

struct OutputParameters {
    std::string OutputFilename;
};

class ConfigParser {
public:
    ConfigParser(const std::string& filename, Logger& logger);
    bool parse();

    DataSourceParameters getDataSourceParameters() const;
    SimulationParameters getSimulationParameters() const;
    // CRITICAL: Ensure ModelParameters is correctly used here
    ModelParameters getModelParameters() const;
    FilterParameters getFilterParameters() const;
    OutputParameters getOutputParameters() const;

    std::string getPythonScriptPath() const;
    std::string getPythonFunctionName() const;
    std::string getTicker() const;
    std::string getStartDate() const;
    std::string getEndDate() const;
    std::string getTempOutputCSVPath() const;

    std::string getLoggingFilename() const;
    Logger::LogLevel getLoggingLevelThreshold() const;

private:
    std::string config_filename;
    std::map<std::string, std::map<std::string, std::string>> config_map;
    Logger& logger;

    DataSourceParameters ds_params;
    SimulationParameters sim_params;
    // CRITICAL: Ensure ModelParameters is correctly used here
    ModelParameters model_params;
    FilterParameters filter_params;
    OutputParameters output_params;

    std::string python_script_path;
    std::string python_function_name;
    std::string ticker;
    std::string start_date;
    std::string end_date;
    std::string temp_output_csv_path;

    std::string logging_filename;
    Logger::LogLevel logging_level_threshold;

    bool getBool(const std::string& section, const std::string& key, bool default_val);
    int getInt(const std::string& section, const std::string& key, int default_val);
    double getDouble(const std::string& section, const std::string& key, double default_val);
    std::string getString(const std::string& section, const std::string& key, const std::string& default_val);

    void parseLogLevel(const std::string& level_str);
};

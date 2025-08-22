// Data/DataReader.h
#pragma once

#include <string>
#include <vector>
// Custom headers (paths relative to project root)
#include "Utils/Logger.h" // Logger class

class DataReader {
public:
    DataReader(const std::string& filename, int return_column_index, Logger& logger);
    std::vector<double> readData();

private:
    std::string data_filename;
    int return_col_idx;
    Logger& logger;
};

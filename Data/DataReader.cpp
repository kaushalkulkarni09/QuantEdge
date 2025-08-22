// Data/DataReader.cpp
#include "Data/DataReader.h" // Self-include
#include <fstream>
#include <sstream>
#include <string>
#include <limits> // For numeric_limits

DataReader::DataReader(const std::string& filename, int return_column_index, Logger& logger)
    : data_filename(filename), return_col_idx(return_column_index), logger(logger) {}

std::vector<double> DataReader::readData() {
    std::vector<double> data_points;
    std::ifstream file(data_filename);

    if (!file.is_open()) {
        logger.log(Logger::ERROR, "Could not open data file: %s", data_filename.c_str());
        return data_points;
    }

    std::string line;
    // Skip header line
    if (std::getline(file, line)) {
        logger.log(Logger::INFO, "Skipping header: %s", line.c_str());
    } else {
        logger.log(Logger::WARNING, "Data file is empty or only contains a header.");
        return data_points;
    }

    int line_num = 1; // Start from after header
    while (std::getline(file, line)) {
        line_num++;
        std::stringstream ss(line);
        std::string cell;
        int current_col = 0;
        bool data_read_for_line = false;

        while (std::getline(ss, cell, ',')) { // Assuming comma-separated values
            if (current_col == return_col_idx) {
                try {
                    // Check if cell is empty or just whitespace
                    std::string trimmed_cell;
                    size_t first = cell.find_first_not_of(" \t\n\r\f\v");
                    if (std::string::npos == first) { // Empty or all whitespace
                        throw std::invalid_argument("Empty or whitespace-only cell");
                    }
                    size_t last = cell.find_last_not_of(" \t\n\r\f\v");
                    trimmed_cell = cell.substr(first, (last - first + 1));

                    if (trimmed_cell.empty()) {
                        throw std::invalid_argument("Empty cell after trim");
                    }

                    double value = std::stod(trimmed_cell);
                    data_points.push_back(value);
                    data_read_for_line = true;
                    break; // Found the desired column, move to next line
                } catch (const std::invalid_argument& e) {
                    logger.log(Logger::WARNING, "Error parsing line %d, column %d: '%s' - %s", line_num, return_col_idx, cell.c_str(), e.what());
                    data_read_for_line = false; // Mark that this line's data was not read successfully
                    break; // Skip to next line
                } catch (const std::out_of_range& e) {
                    logger.log(Logger::WARNING, "Value out of range in line %d, column %d: '%s' - %s", line_num, return_col_idx, cell.c_str(), e.what());
                    data_read_for_line = false;
                    break;
                }
            }
            current_col++;
        }
        if (!data_read_for_line && !line.empty() && line.find_first_not_of(" \t\n\r\f\v,") != std::string::npos) {
            logger.log(Logger::WARNING, "No valid data found in column %d on line %d. Line content: '%s'", return_col_idx, line_num, line.c_str());
        }
    }
    file.close();
    return data_points;
}

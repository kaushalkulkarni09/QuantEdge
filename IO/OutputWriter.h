// IO/OutputWriter.h
#pragma once

#include <string>
#include <fstream>
// Custom headers (paths relative to project root)
#include "Utils/Logger.h" // Logger class

class OutputWriter {
public:
    OutputWriter(const std::string& filename, Logger& logger);
    ~OutputWriter();

    bool writeLine(const std::string& line);

private:
    std::string output_filename;
    std::ofstream output_file;
    Logger& logger;
};

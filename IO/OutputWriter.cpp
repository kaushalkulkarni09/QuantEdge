// IO/OutputWriter.cpp
#include "IO/OutputWriter.h" // Self-include
#include <iostream>

OutputWriter::OutputWriter(const std::string& filename, Logger& logger)
    : output_filename(filename), logger(logger) {
    output_file.open(output_filename);
    if (!output_file.is_open()) {
        logger.log(Logger::ERROR, "Failed to open output file: %s", output_filename.c_str());
    }
}

OutputWriter::~OutputWriter() {
    if (output_file.is_open()) {
        output_file.close();
    }
}

bool OutputWriter::writeLine(const std::string& line) {
    if (output_file.is_open()) {
        output_file << line << std::endl;
        return true;
    }
    return false;
}

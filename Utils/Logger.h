// Utils/Logger.h
#pragma once

#include <string>
#include <fstream>
#include <mutex> // For multithreading safety
#include <iomanip> // For std::put_time

class Logger {
public:
    enum LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR
    };

    Logger(const std::string& filename, LogLevel level);
    ~Logger();

    // Log a message with a specific level and format string (like printf)
    void log(LogLevel level, const char* format, ...) __attribute__((format(printf, 3, 4)));

    // Reinitialize logger settings from config
    void reinitialize(const std::string& new_filename, LogLevel new_level);

    // Convert LogLevel enum to string for output
    static std::string levelToString(LogLevel level);

private:
    std::string log_filename;
    std::ofstream log_file;
    LogLevel log_level_threshold;
    std::mutex log_mutex; // Mutex for thread-safe logging

    // Get current timestamp for log entries
    std::string getCurrentTimestamp();
};

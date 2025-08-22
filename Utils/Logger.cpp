// Utils/Logger.cpp
#include "Utils/Logger.h" // Self-include
#include <cstdarg> // For va_list, va_start, va_end
#include <chrono>  // For std::chrono::system_clock
#include <ctime>   // For std::time_t, std::localtime, std::mktime
#include <iomanip> // For std::put_time
#include <iostream> // For std::cerr in case of file open error
#include <sstream> // <<< ADD THIS LINE for std::stringstream

// Constructor: Opens log file and sets initial log level.
Logger::Logger(const std::string& filename, LogLevel level)
    : log_filename(filename), log_level_threshold(level) {
    log_file.open(log_filename, std::ios::app); // Open in append mode
    if (!log_file.is_open()) {
        std::cerr << "Warning: Could not open log file: " << log_filename << ". Logging to console only." << std::endl;
    }
}

// Destructor: Closes the log file.
Logger::~Logger() {
    if (log_file.is_open()) {
        log_file.close();
    }
}

// Reinitializes the logger with new filename and log level.
void Logger::reinitialize(const std::string& new_filename, LogLevel new_level) {
    std::lock_guard<std::mutex> lock(log_mutex); // Ensure thread safety

    // Close old file if open
    if (log_file.is_open()) {
        log_file.close();
    }

    log_filename = new_filename;
    log_level_threshold = new_level;

    // Open new file
    log_file.open(log_filename, std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Warning: Could not open log file: " << log_filename << " after reinitialization. Logging to console only." << std::endl;
    }
}

// Logs a message if its level meets the threshold.
void Logger::log(LogLevel level, const char* format, ...) {
    if (level < log_level_threshold) {
        return; // Message level is below the threshold, so don't log
    }

    std::lock_guard<std::mutex> lock(log_mutex); // Ensure thread safety for logging

    std::string timestamp = getCurrentTimestamp();
    std::string level_str = levelToString(level);

    // Format the message
    char buffer[1024]; // A fixed-size buffer for the formatted message
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    // Construct the full log message
    std::string log_message = "[" + timestamp + "] [" + level_str + "] " + buffer;

    // Log to console (stderr for ERROR, stdout for others)
    if (level == ERROR) {
        std::cerr << log_message << std::endl;
    } else {
        std::cout << log_message << std::endl;
    }

    // Log to file if it's open
    if (log_file.is_open()) {
        log_file << log_message << std::endl;
    }
}

// Gets the current timestamp in a formatted string.
std::string Logger::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t current_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&current_time); // Use localtime for local timezone

    std::stringstream ss; // This line now has std::stringstream available
    ss << std::put_time(local_time, "%Y-%m-%d %H:%M:%S"); // Format:YYYY-MM-DD HH:MM:SS
    return ss.str();
}

// Converts LogLevel enum to string.
std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case DEBUG: return "DEBUG";
        case INFO: return "INFO";
        case WARNING: return "WARNING";
        case ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

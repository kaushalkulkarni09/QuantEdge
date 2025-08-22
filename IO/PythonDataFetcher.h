
#pragma once

#include <string>

#include <Python.h>

class PythonDataFetcher {
public:
    PythonDataFetcher(const std::string& python_script_path,
                      const std::string& python_function_name,
                      const std::string& temp_output_csv_path);
    ~PythonDataFetcher();

    bool initialize();
    bool fetchData(const std::string& ticker, const std::string& start_date, const std::string& end_date);


    bool isInitialized() const { return is_initialized; }


    void printError() { printPythonError(); }

private:
    std::string script_path;
    std::string function_name;
    std::string temp_csv_path;

    PyObject* pModule;
    PyObject* pFunc;
    bool is_initialized; // Tracks if Py_Initialize() has been called successfully


    void printPythonError();
};

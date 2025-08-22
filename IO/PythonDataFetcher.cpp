
#include "IO/PythonDataFetcher.h"
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <filesystem>

PythonDataFetcher::PythonDataFetcher(const std::string& python_script_path,
                                     const std::string& python_function_name,
                                     const std::string& temp_output_csv_path)
    : script_path(python_script_path),
      function_name(python_function_name),
      temp_csv_path(temp_output_csv_path),
      pModule(nullptr),
      pFunc(nullptr),
      is_initialized(false) {
}

PythonDataFetcher::~PythonDataFetcher() {
    if (pFunc) {
        Py_DECREF(pFunc);
    }
    if (pModule) {
        Py_DECREF(pModule);
    }
    if (is_initialized) {
        Py_Finalize();
    }
}

void PythonDataFetcher::printPythonError() {
    if (PyErr_Occurred()) {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);

        PyObject* pStr = PyObject_Str(pvalue);
        if (pStr) {
            std::cerr << "Python Error: " << PyUnicode_AsUTF8(pStr) << std::endl;
            Py_DECREF(pStr);
        }
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
    }
}


bool PythonDataFetcher::initialize() {
    if (is_initialized) {
        std::cout << "Python interpreter already initialized." << std::endl;
        return true;
    }

    Py_SetProgramName(L"MathProject"); //
    Py_Initialize();
    if (!Py_IsInitialized()) {
        std::cerr << "Error: Failed to initialize Python interpreter." << std::endl;
        return false;
    }

    PyObject* sysPath = PySys_GetObject("path");
    if (!sysPath) {
        std::cerr << "Error: Could not get Python sys.path." << std::endl;
        printPythonError();
        Py_Finalize();
        return false;
    }

    // Add script directory to Python path
    size_t last_slash_pos = script_path.find_last_of("/\\");
    std::string script_dir = (last_slash_pos == std::string::npos) ? "." : script_path.substr(0, last_slash_pos);
    PyObject* py_script_path = PyUnicode_DecodeFSDefault(script_dir.c_str());
    if (PyList_Append(sysPath, py_script_path) < 0) {
        std::cerr << "Error: Could not add script directory to Python path." << std::endl;
        printPythonError();
        Py_XDECREF(py_script_path);
        Py_Finalize();
        return false;
    }
    Py_XDECREF(py_script_path);


    const std::string site_packages_path = "/Users/kaushalkulkarni/CLionProjects/MathProject/.venv/lib/python3.9/site-packages";
    PyObject* py_site_packages_path = PyUnicode_DecodeFSDefault(site_packages_path.c_str());
    if (PyList_Append(sysPath, py_site_packages_path) < 0) {
        std::cerr << "Error: Could not add site-packages directory to Python path: " << site_packages_path << std::endl;
        printPythonError();
        Py_XDECREF(py_site_packages_path);
        Py_Finalize();
        return false;
    }
    Py_XDECREF(py_site_packages_path);
    std::cout << "Added site-packages to sys.path: " << site_packages_path << std::endl;

    // Load the Python module (e.g., "download_data" from "download_data.py")
    std::string module_name_str = script_path.substr(last_slash_pos == std::string::npos ? 0 : last_slash_pos + 1);
    if (module_name_str.length() > 3 && module_name_str.substr(module_name_str.length() - 3) == ".py") {
        module_name_str = module_name_str.substr(0, module_name_str.length() - 3);
    }

    pModule = PyImport_ImportModule(module_name_str.c_str());
    if (!pModule) {
        std::cerr << "Error: Failed to load Python module: '" << module_name_str << "'" << std::endl;
        printPythonError();
        Py_Finalize();
        return false;
    }

    // Get the function from the module
    pFunc = PyObject_GetAttrString(pModule, function_name.c_str());
    if (!pFunc || !PyCallable_Check(pFunc)) {
        if (PyErr_Occurred()) printPythonError();
        std::cerr << "Error: Cannot find function '" << function_name << "' in module '" << module_name_str << "' or it is not callable." << std::endl;
        Py_XDECREF(pModule);
        Py_Finalize(); // Corrected from Py_FinalFinit()
        return false;
    }

    is_initialized = true;
    std::cout << "PythonDataFetcher initialized successfully." << std::endl;
    return true;
}

bool PythonDataFetcher::fetchData(const std::string& ticker, const std::string& start_date, const std::string& end_date) {
    if (!is_initialized) {
        std::cerr << "Error: PythonDataFetcher not initialized. Call initialize() first." << std::endl;
        return false;
    }

    PyObject* pArgs = PyTuple_New(4); // 4 arguments: ticker, start_date, end_date, output_csv_path
    if (!pArgs) {
        printPythonError();
        return false;
    }

    // Convert C++ strings to Python Unicode objects
    PyObject* py_ticker = PyUnicode_FromString(ticker.c_str());
    PyObject* py_start_date = PyUnicode_FromString(start_date.c_str());
    PyObject* py_end_date = PyUnicode_FromString(end_date.c_str());
    PyObject* py_temp_csv_path = PyUnicode_FromString(temp_csv_path.c_str());

    if (!py_ticker || !py_start_date || !py_end_date || !py_temp_csv_path) {
        std::cerr << "Error: Failed to convert C++ strings to Python objects." << std::endl;
        Py_XDECREF(py_ticker); Py_XDECREF(py_start_date); Py_XDECREF(py_end_date); Py_XDECREF(py_temp_csv_path);
        Py_XDECREF(pArgs);
        printPythonError();
        return false;
    }

    // Put arguments into the tuple
    PyTuple_SetItem(pArgs, 0, py_ticker); // PyTuple_SetItem "steals" a reference
    PyTuple_SetItem(pArgs, 1, py_start_date);
    PyTuple_SetItem(pArgs, 2, py_end_date);
    PyTuple_SetItem(pArgs, 3, py_temp_csv_path);


    // Call the Python function
    PyObject* pValue = PyObject_CallObject(pFunc, pArgs);


    Py_DECREF(pArgs);

    if (pValue == nullptr) {
        std::cerr << "Error: Call to Python function '" << function_name << "' failed." << std::endl;
        printPythonError();
        return false;
    }

    // Check return value (assuming Python function returns a boolean success indicator)
    bool success = (pValue == Py_True);
    Py_DECREF(pValue);

    if (!success) {
        std::cerr << "Python function '" << function_name << "' reported an error or non-success." << std::endl;
    }
    return success;
}

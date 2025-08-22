// Utils/MathUtils.h
#pragma once

#include <cmath> // For std::sqrt, etc.

// This file can be used for general mathematical utilities if needed.

namespace MathUtils {
    // Example: A simple utility function
    inline double safe_sqrt(double val) {
        if (val < 0) return 0.0; // Or throw an error, depending on desired behavior
        return std::sqrt(val);
    }


}

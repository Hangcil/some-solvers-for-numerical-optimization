#ifndef MINI_OPTIM_FEASIBLE_DIRECTION
#define MINI_OPTIM_FEASIBLE_DIRECTION

#include "linear_algebra.h"
#include "line_programming.h"
#include "line_search.h"
#include <vector>

namespace mini_optim
{
    class zoutendijk_i
    {
    public:
        obj_fun f;
        std::vector<std::vector<double>> A, Aeq;
        std::vector<double> x0, b, beq;
        unsigned long long max_iter = 1000;
    };
    class zoutendijk_o
    {
    public:
        bool success;
        unsigned iter_num;
        double obj_value;
        std::vector<double> result;
    };
    zoutendijk_o zoutendijk(const zoutendijk_i &param);
}

#endif
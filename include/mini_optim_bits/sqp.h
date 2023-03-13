#ifndef MINI_OPTIM_SQP
#define MINI_OPTIM_SQP

#include "linear_algebra.h"
#include "quadratic_programming.h"
#include "line_search.h"
#include <numeric>

using nonlin_con = std::function<std::vector<double>(std::vector<double> &)>;

namespace mini_optim
{
    class sqp_i
    {
    public:
        std::vector<std::vector<double>> A, Aeq;
        std::vector<double> b, beq, x0;
        unsigned max_iter = 200;
        double tol = 0.000000001;
        obj_fun f;
    };
    class sqp_o
    {
    public:
        double obj_value;
        unsigned iter = 0;
        bool success;
        std::vector<double> result;
    };
    sqp_o sqp(const sqp_i &input);
}
#endif
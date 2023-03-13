#ifndef MINI_OPTIM_QUADRATIC_PROGRAMMING
#define MINI_OPTIM_QUADRATIC_PROGRAMMING

#include "linear_algebra.h"
#include "line_programming.h"
#include <algorithm>

namespace mini_optim
{
    class qp_i
    {
    public:
        std::vector<std::vector<double>> H, A, Aeq;
        std::vector<double> c, b, beq, x0;
    };
    class qp_o
    {
    public:
        double obj_value;
        unsigned iter = 0;
        bool success;
        std::vector<double> result, multiplier;
    };
    qp_o lagrange(const qp_i &input);
    qp_o active_set(const qp_i &input);
    qp_o quadprog(const qp_i &input);
}
#endif
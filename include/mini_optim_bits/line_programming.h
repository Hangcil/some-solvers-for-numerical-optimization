#ifndef MINI_OPTIM_LINE_PROGRAMMING
#define MINI_OPTIM_LINE_PROGRAMMING

#include "linear_algebra.h"
#include <vector>
#include <algorithm>
namespace mini_optim
{
    class linprog_o
    {
    public:
        bool success;
        unsigned iter_num;
        double obj_value;
        std::vector<unsigned> base_ind;
        std::vector<double> result;
    };
    class linprog_i
    {
    public:
        std::vector<std::vector<double>> A, Aeq;
        std::vector<double> c, b, beq;
    };
    linprog_o simplex_core(const std::vector<double> &c, const std::vector<std::vector<double>> &A, const std::vector<double> &b, const std::vector<unsigned> &ind);
    linprog_o simplex(const std::vector<double> &c, const std::vector<std::vector<double>> &A, const std::vector<double> &b);
    linprog_o linprog(const linprog_i &param_input);
}

#endif
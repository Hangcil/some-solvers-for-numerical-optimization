#ifndef MINI_OPTIM_LINE_SEARCH
#define MINI_OPTIM_LINE_SEARCH

#include <vector>
#include <functional>
#include "linear_algebra.h"

using obj_fun = std::function<double(std::vector<double> &)>;

namespace mini_optim
{
    std::vector<double> line_search(const obj_fun &f, const std::vector<double> &a, const std::vector<double> &b);
    std::vector<std::vector<double>> min_range(const obj_fun &f, const std::vector<double> &x0, const std::vector<double> &dir);
    std::vector<double> grad(const obj_fun &f, const std::vector<double> &pos);
}

#endif
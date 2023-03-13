#ifndef MINI_OPTIM_LINEAR_ALGEBRA
#define MINI_OPTIM_LINEAR_ALGEBRA

#include <vector>
#include <cmath>

namespace mini_optim
{
    double dot(const std::vector<double> &v1, const std::vector<double> &v2);
    std::vector<double> add(const std::vector<double> &v1, const std::vector<double> &v2);
    std::vector<std::vector<double>> add(const std::vector<std::vector<double>> &A1, const std::vector<std::vector<double>> &A2);
    std::vector<double> multiply(double a, const std::vector<double> &v);
    std::vector<double> multiply(const std::vector<std::vector<double>> &A, const std::vector<double> &v);
    std::vector<std::vector<double>> multiply(double a, const std::vector<std::vector<double>> &A);
    std::vector<std::vector<double>> multiply(const std::vector<std::vector<double>> &A1, const std::vector<std::vector<double>> &A2);
    std::vector<std::vector<double>> multiply(const std::vector<double> &v1, const std::vector<double> &v2);
    std::vector<std::vector<double>> cholesky(const std::vector<std::vector<double>> &A);
    std::vector<double> solve(const std::vector<std::vector<double>> &A, const std::vector<double> &b);
    std::vector<std::vector<double>> trans(const std::vector<std::vector<double>> &A);
    std::vector<std::vector<double>> eye(unsigned n);
}

#endif
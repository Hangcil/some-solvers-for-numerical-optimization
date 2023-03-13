#include "../../include/mini_optim_bits/line_search.h"

using namespace mini_optim;

std::vector<std::vector<double>> mini_optim::min_range(const obj_fun &f, const std::vector<double> &x0, const std::vector<double> &dir)
{
    double h = 0.01;
    std::vector<double> x1 = x0, x2, x3, x4;
    double f1 = f(x1), f4;
    int k = 0;
    long long iter = 0;
    while (true)
    {
        iter++;
        x4 = add(x1, multiply(h, dir));
        f4 = f(x4);
        k++;
        if (f4 < f1)
        {
            x2 = x1;
            x1 = x4;
            f1 = f4;
            h *= 1.5;
            continue;
        }
        else
        {
            if (k == 1 && iter < 50)
            {
                h *= -1;
                x2 = x4;
                continue;
            }
            else
            {
                x3 = x2;
                x1 = x4;
                std::vector<std::vector<double>> result;
                result.push_back(x1);
                result.push_back(x3);
                return result;
            }
        }
    }
}

std::vector<double> mini_optim::line_search(const obj_fun &f, const std::vector<double> &a, const std::vector<double> &b)
{
    auto g = [&](double x) -> double
    {
        std::vector<double> temp = add(multiply(1 - x, a), multiply(x, b));
        return f(temp);
    };
    double a1 = 0.0, b1 = 1.0, l = 0.001,
           lambda = a1 + 0.382 * (b1 - a1), mu = a1 + 0.618 * (b1 - a1);
    long long ite = 0;
    while (true)
    {
        ite++;
        if (b1 - a1 <= l || ite >= 50)
        {
            std::vector<double> result = add(multiply((1 - (b1 + a1) / 2), a), multiply((b1 + a1) / 2, b));
            return result;
        }
        else
        {
            if (g(lambda) > g(mu))
            {
                a1 = lambda;
                lambda = mu;
                mu = a1 + 0.618 * (b1 - a1);
                continue;
            }
            else
            {
                b1 = mu;
                mu = lambda;
                lambda = a1 + 0.382 * (b1 - a1);
                continue;
            }
        }
    }
}

std::vector<double> mini_optim::grad(const obj_fun &f, const std::vector<double> &pos)
{
    double alpha = 0.9, eps = 0.0001;
    auto nvar = pos.size();
    std::vector<double> result(nvar, 0);
    for (auto i = 0; i < nvar; i++)
    {
        auto par = [&](double x) -> double
        {
            std::vector<double> x1 = pos;
            x1[i] += x;
            return f(x1);
        };
        double h = 0.1;
        double p = (par(h) - par(-h)) / 2 / h;
        long long ite = 0;
        while (true)
        {
            ite++;
            h *= alpha;
            double p_ = (-par(h / 2) + 8 * par(h / 4) - 8 * par(-h / 4) + par(-h / 2)) / 3 / h;
            if (std::abs(p - p_) <= eps || ite >= 50)
            {
                result[i] = p_;
                break;
            }
            else
            {
                p = p_;
            }
        }
    }
    return result;
}
#include "../../include/mini_optim_bits/linear_algebra.h"

using namespace mini_optim;

double mini_optim::dot(const std::vector<double> &v1, const std::vector<double> &v2)
{
    unsigned m = v1.size(), n = v2.size();
    if (m != n)
    {
        return 0;
    }
    else
    {
        double sum = 0;
        for (unsigned i = 0; i < m; i++)
        {
            sum += v1[i] * v2[i];
        }
        return sum;
    }
}

std::vector<double> mini_optim::multiply(double a, const std::vector<double> &v)
{
    std::vector<double> result;
    for (auto i : v)
    {
        result.push_back(a * i);
    }
    return result;
}

std::vector<double> mini_optim::add(const std::vector<double> &v1, const std::vector<double> &v2)
{
    std::vector<double> r;
    if (v1.size() == v2.size())
    {
        for (auto i = 0; i < v1.size(); i++)
        {
            r.push_back(v1[i] + v2[i]);
        }
    }
    return r;
}

std::vector<std::vector<double>> mini_optim::add(const std::vector<std::vector<double>> &A1, const std::vector<std::vector<double>> &A2)
{
    std::vector<std::vector<double>> r;
    for (auto i = 0; i < A1.size(); i++)
    {
        r.push_back(add(A1[i], A2[i]));
    }
    return r;
}

std::vector<double> mini_optim::multiply(const std::vector<std::vector<double>> &A, const std::vector<double> &v)
{
    std::vector<double> r;
    for (auto i : A)
    {
        r.push_back(dot(i, v));
    }
    return r;
}

std::vector<std::vector<double>> mini_optim::multiply(double a, const std::vector<std::vector<double>> &A)
{
    std::vector<std::vector<double>> r = A;
    for (auto i = 0; i < A.size(); i++)
    {
        for (auto j = 0; j < A[i].size(); j++)
        {
            r[i][j] = a * A[i][j];
        }
    }
    return r;
}

std::vector<std::vector<double>> mini_optim::multiply(const std::vector<std::vector<double>> &A1, const std::vector<std::vector<double>> &A2)
{
    std::vector<std::vector<double>> ret(A1.size());
    for (auto i = 0; i < A2[0].size(); i++)
    {
        std::vector<double> col_A2;
        for (auto j = 0; j < A2.size(); j++)
        {
            col_A2.push_back(A2[j][i]);
        }
        auto col_ret = multiply(A1, col_A2);
        for (auto j = 0; j < ret.size(); j++)
        {
            ret[j].push_back(col_ret[j]);
        }
    }
    return ret;
}

std::vector<std::vector<double>> mini_optim::multiply(const std::vector<double> &v1, const std::vector<double> &v2)
{
    std::vector<std::vector<double>> r;
    for (auto i = 0; i < v1.size(); i++)
    {
        std::vector<double> row;
        for (auto j = 0; j < v1.size(); j++)
        {
            row.push_back(v1[i] * v2[j]);
        }
        r.push_back(row);
    }
    return r;
}

std::vector<std::vector<double>> mini_optim::trans(const std::vector<std::vector<double>> &A)
{
    std::vector<std::vector<double>> ret(A[0].size());
    for (auto i = 0; i < A.size(); i++)
    {
        for (auto j = 0; j < ret.size(); j++)
        {
            ret[j].push_back(A[i][j]);
        }
    }
    return ret;
}

std::vector<std::vector<double>> mini_optim::eye(unsigned n)
{
    std::vector<std::vector<double>> ret;
    for (auto i = 0; i < n; i++)
    {
        std::vector<double> col(n, 0.0);
        col[i] = 1.0;
        ret.push_back(col);
    }
    return ret;
}

std::vector<std::vector<double>> mini_optim::cholesky(const std::vector<std::vector<double>> &A)
{
    auto r = A;
    auto n = r.size();
    for (auto i = 0; i < n; i++)
    {
        r[i][i] = sqrt(r[i][i]);
        if (i < n - 1)
        {
            for (auto j = i + 1; j < n; j++)
            {
                r[j][i] /= r[i][i];
            }
        }
        for (auto j = i + 1; j < n; j++)
        {
            for (auto k = j; k < n; k++)
            {
                r[k][j] = r[k][j] - r[k][i] * r[j][i];
            }
        }
    }
    return r;
}

std::vector<double> mini_optim::solve(const std::vector<std::vector<double>> &A, const std::vector<double> &b)
{
    auto H = multiply(trans(A), A);
    auto L = cholesky(H);
    auto c = multiply(trans(A), b);
    std::vector<double> y, x;
    for (auto i = 0; i < A.size() - 1; i++)
    {
        c[i] /= L[i][i];
        for (auto j = i + 1; j < A.size(); j++)
        {
            c[j] -= c[i] * L[j][i];
        }
    }
    c[A.size() - 1] /= L[A.size() - 1][A.size() - 1];
    for (auto i = A.size() - 1; i >= 1; i--)
    {
        c[i] /= L[i][i];
        for (auto j = i - 1;; j--)
        {
            c[j] -= c[i] * L[i][j];
            if (j <= 0)
            {
                break;
            }
        }
    }
    c[0] /= L[0][0];
    return c;
}

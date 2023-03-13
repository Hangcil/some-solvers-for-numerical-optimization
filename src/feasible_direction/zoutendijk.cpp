#include "../../include/mini_optim_bits/feasible_direction.h"

using namespace mini_optim;

zoutendijk_o mini_optim::zoutendijk(const zoutendijk_i &param)
{
    double zero = 0.00000000001;
    auto A = param.A;
    auto b = param.b, x = param.x0, save = x;
    unsigned iter = 0;
    while (true)
    {
        iter++;
        std::vector<std::vector<double>> A1, A2;
        std::vector<double> b1, b2;
        auto test = add(multiply(-1.0, b), multiply(A, x));
        for (auto i = 0; i < test.size(); i++)
        {
            if (test[i] <= zero)
            {
                A1.push_back(A[i]);
                b1.push_back(b[i]);
            }
            else
            {
                A2.push_back(A[i]);
                b2.push_back(b[i]);
            }
        }
        auto A_ = multiply(-1.0, A1);
        std::vector<std::vector<double>> E;
        for (auto i = 0; i < x.size(); i++)
        {
            std::vector<double> temp(x.size(), 0);
            temp[i] = 1.0;
            E.push_back(temp);
        }
        auto _E = multiply(-1.0, E);
        A_.insert(A_.end(), E.begin(), E.end());
        A_.insert(A_.end(), _E.begin(), _E.end());
        std::vector<double> b_(A1.size(), 0), ones(x.size() * 2, 1.0), zeros(param.Aeq.size(), 0);
        b_.insert(b_.end(), ones.begin(), ones.end());
        linprog_i in;
        in.c = grad(param.f, x);
        in.A = A_;
        in.b = b_;
        in.Aeq = param.Aeq;
        in.beq = zeros;
        auto r = linprog(in);
        if (!r.success || r.result.empty())
        {
            zoutendijk_o result;
            result.success = false;
            result.iter_num = iter;
            return result;
        }
        else
        {
            double lambda_m = 0;
            auto _b = add(b2, multiply(-1.0, multiply(A2, x)));
            auto _d = multiply(A2, r.result);
            std::vector<double> lamb_candidate;
            for (auto i = 0; i < _d.size(); i++)
            {
                if (_d[i] < 0.0)
                {
                    lamb_candidate.push_back(_b[i] / _d[i]);
                }
            }
            save = x;
            if (lamb_candidate.empty())
            {
                auto range = min_range(param.f, x, r.result);
                x = line_search(param.f, range[0], range[1]);
            }
            else
            {
                lambda_m = *std::min_element(lamb_candidate.begin(), lamb_candidate.end());
                double va = param.f(x);
                x = line_search(param.f, x, add(x, multiply(lambda_m, r.result)));
            }
            auto prog = add(multiply(-1.0, save), x);
            double sum = 0;
            for (auto i : prog)
            {
                sum += std::abs(i);
            }
            if (std::abs(dot(grad(param.f, x), r.result)) <= 100 * zero || iter >= param.max_iter || sum <= 100 * zero)
            {
                zoutendijk_o result;
                result.success = true;
                result.result = x;
                result.obj_value = param.f(x);
                result.iter_num = iter;
                return result;
            }
        }
    }
}
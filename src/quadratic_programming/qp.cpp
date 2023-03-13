#include "../../include/mini_optim_bits/quadratic_programming.h"

using namespace mini_optim;
using namespace std;

qp_o mini_optim::lagrange(const qp_i &input)
{
    if (input.Aeq.size() != input.beq.size() || input.H.empty())
    {
        qp_o ret;
        ret.success = false;
        return ret;
    }
    bool is_mat = true;
    if (input.Aeq.empty())
    {
        qp_o ret;
        ret.success = true;
        auto x = solve(input.H, multiply(-1.0, input.c));
        ret.result = x;
        ret.obj_value = 0.5 * dot(x, multiply(input.H, x)) + dot(input.c, x);
        return ret;
    }
    else
    {
        auto n = input.Aeq[0].size();
        for (auto i = 0; i < input.Aeq.size(); i++)
        {
            is_mat = is_mat && input.Aeq[i].size() == n;
        }
        is_mat = is_mat && input.H[0].size() == n;
        is_mat = is_mat && input.c.size() == n;
        is_mat = is_mat && input.H.size() == n;
        if (!is_mat)
        {
            qp_o ret;
            ret.success = false;
            return ret;
        }
    }
    auto H_ = input.H;
    auto _A = multiply(-1.0, input.Aeq);
    H_.insert(H_.end(), _A.begin(), _A.end());
    for (auto i = 0; i < H_.size(); i++)
    {
        if (i < input.H.size())
        {
            for (auto j = 0; j < input.Aeq.size(); j++)
            {
                H_[i].push_back(-1.0 * input.Aeq[j][i]);
            }
        }
        else
        {
            for (auto j = 0; j < input.Aeq.size(); j++)
            {
                H_[i].push_back(0.0);
            }
        }
    }
    auto c = input.c;
    c.insert(c.end(), input.beq.begin(), input.beq.end());
    auto x = solve(H_, multiply(-1.0, c));
    if (x.empty())
    {
        qp_o ret;
        ret.success = false;
        return ret;
    }
    qp_o ret;
    ret.success = true;
    std::vector<double> multiplier;
    for (auto i = 0; i < x.size() - input.c.size(); i++)
    {
        multiplier.push_back(x[i + input.c.size()]);
    }
    x.resize(input.c.size());
    ret.result = x;
    ret.multiplier = multiplier;
    ret.obj_value = 0.5 * dot(x, multiply(input.H, x)) + dot(input.c, x);
    return ret;
}

qp_o mini_optim::active_set(const qp_i &input)
{
    if ((input.Aeq.empty() && input.A.empty()) || (input.beq.empty() && input.b.empty()) || input.Aeq.size() != input.beq.size() || input.H.empty() || input.A.size() != input.b.size())
    {
        qp_o ret;
        ret.success = false;
        return ret;
    }
    bool is_mat = true;
    unsigned n = 0;
    if (!input.Aeq.empty())
    {
        n = input.Aeq[0].size();
        for (auto i = 0; i < input.Aeq.size(); i++)
        {
            is_mat = is_mat && input.Aeq[i].size() == n;
        }
    }
    if (!input.A.empty())
    {
        n = input.A[0].size();
        for (auto i = 0; i < input.A.size(); i++)
        {
            is_mat = is_mat && input.A[i].size() == n;
        }
    }
    is_mat = is_mat && input.H[0].size() == n;
    is_mat = is_mat && input.c.size() == n;
    is_mat = is_mat && input.H.size() == n;
    if (!is_mat)
    {
        qp_o ret;
        ret.success = false;
        return ret;
    }
    unsigned iter = 0;
    auto x = input.x0;
    double zero = 0.0000000001;
    std::vector<unsigned> active_i;
    auto test = add(multiply(-1.0, input.b), multiply(input.A, x));
    for (auto i = 0; i < test.size(); i++)
    {
        if (std::abs(test[i]) <= zero)
        {
            active_i.push_back(i);
        }
    }
    std::vector<double> delta;
    while (true)
    {
        iter++;
        std::vector<std::vector<double>> A_;
        for (auto i : active_i)
        {
            A_.push_back(input.A[i]);
        }
        auto c_ = add(multiply(input.H, x), input.c);
        std::vector<double> b_(active_i.size(), 0.0);
        qp_i in;
        in.Aeq = A_;
        in.H = input.H;
        in.beq = b_;
        in.c = c_;
        auto r = lagrange(in);
        delta = r.result;
        if (dot(delta, delta) > zero)
        {
            std::vector<double> temp;
            std::vector<unsigned> A_ind;
            for (auto i = 0; i < input.A.size(); i++)
            {
                if (find(active_i.begin(), active_i.end(), (unsigned)i) == active_i.end() && dot(input.A[i], delta) < 0.0)
                {
                    temp.push_back((input.b[i] - dot(input.A[i], x)) / dot(input.A[i], delta));
                    A_ind.push_back(i);
                }
            }
            if (!temp.empty())
            {
                unsigned ind = min_element(temp.begin(), temp.end()) - temp.begin();
                auto alpha = temp[ind];
                auto p = A_ind[ind];
                alpha = alpha >= 1.0 ? 1.0 : alpha;
                x = add(x, multiply(alpha, delta));
                if (alpha - 1.0 < 0.0)
                {
                    active_i.push_back(p);
                    continue;
                }
                else
                {
                    active_i.clear();
                    auto test1 = add(multiply(-1.0, input.b), multiply(input.A, x));
                    for (auto i = 0; i < test1.size(); i++)
                    {
                        if (std::abs(test1[i]) <= zero)
                        {
                            active_i.push_back(i);
                        }
                    }
                }
            }
            else
            {
                x = add(x, delta);
            }
        }
        auto lambda_ = r.multiplier;
        auto min = min_element(lambda_.begin(), lambda_.end());
        if (lambda_.empty() || *min >= 0 || iter >= 100)
        {
            qp_o ret;
            ret.success = true;
            ret.result = x;
            ret.iter = iter;
            std::vector<double> multiplier(input.A.size() + input.Aeq.size(), 0.0);
            for (auto i = 0; i < input.A.size(); i++)
            {
                auto pos = std::find(active_i.begin(), active_i.end(), (unsigned)i);
                if (pos == active_i.end())
                {
                    multiplier[i] = 0.0;
                }
                else
                {
                    multiplier[i] = r.multiplier[*pos];
                }
            }
            for (auto i = 0; i < input.Aeq.size(); i++)
            {
                multiplier[i] = r.multiplier[i + input.A.size()];
            }
            ret.multiplier = multiplier;
            ret.obj_value = 0.5 * dot(x, multiply(input.H, x)) + dot(input.c, x);
            return ret;
        }
        else
        {
            unsigned min_ind = min - lambda_.begin();
            active_i.erase(active_i.begin() + min_ind);
        }
    }
}

qp_o mini_optim::quadprog(const qp_i &input)
{
    qp_i in;
    vector<vector<double>> A;
    vector<double> b;
    for (auto i = 0; i < input.A.size(); i++)
    {
        A.push_back(multiply(-1.0, input.A[i]));
    }
    for (auto i = 0; i < input.b.size(); i++)
    {

        b.push_back(-1.0 * input.b[i]);
    }
    for (auto i = 0; i < input.Aeq.size(); i++)
    {
        A.push_back(input.Aeq[i]);
        A.push_back(multiply(-1.0, input.Aeq[i]));
    }
    for (auto i = 0; i < input.beq.size(); i++)
    {
        b.push_back(input.beq[i]);
        b.push_back(-1.0 * input.beq[i]);
    }
    linprog_i in_;
    vector<double> c(input.c.size(), 0.0);
    auto A_ = input.A, Aeq_ = input.Aeq;
    auto b_ = input.b;
    double zero = 0.000000001;
    for (auto i = 0; i < input.A.size(); i++)
    {
        for (auto j = 0; j < input.A.size(); j++)
        {
            A_[i].push_back(0.0);
        }
        vector<double> temp(input.A.size() + input.c.size(), 0.0);
        A_.push_back(temp);
        b_.push_back(1.0 * zero);
        c.push_back(1.0);
    }
    for (auto i = 0; i < input.Aeq.size(); i++)
    {
        for (auto j = 0; j < input.Aeq.size(); j++)
        {
            Aeq_[i].push_back(0.0);
        }
    }
    for (auto i = 0; i < input.A.size(); i++)
    {
        A_[i][i + input.c.size()] = -1.0;
        A_[i + input.A.size()][i + input.c.size()] = -1.0;
    }
    in_.A = A_;
    in_.Aeq = Aeq_;
    in_.b = b_;
    in_.beq = input.beq;
    in_.c = c;
    auto r0 = linprog(in_);
    auto x0 = r0.result;
    x0.resize(input.c.size());
    in.A = A;
    in.b = b;
    in.H = input.H;
    in.c = input.c;
    in.x0 = x0;
    return active_set(in);
}
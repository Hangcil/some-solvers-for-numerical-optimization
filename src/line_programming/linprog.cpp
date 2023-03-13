#include "../../include/mini_optim_bits/line_programming.h"

using namespace mini_optim;

linprog_o mini_optim::simplex_core(const std::vector<double> &c, const std::vector<std::vector<double>> &A,
                                   const std::vector<double> &b, const std::vector<unsigned> &ind)
{
    unsigned m = c.size(), n = A.size();
    std::vector<double> cb;
    std::vector<double> test_line;
    for (unsigned i = 0; i < n; i++)
    {
        cb.push_back(c[ind[i]]);
    }
    for (unsigned i = 0; i < m; i++)
    {
        std::vector<double> p;
        for (unsigned j = 0; j < n; j++)
        {
            p.push_back(A[j][i]);
        }
        test_line.push_back(dot(cb, p) - c[i]);
    }
    std::vector<std::vector<double>> table = A;
    table.push_back(test_line);
    for (unsigned i = 0; i < n; i++)
    {
        table[i].push_back(b[i]);
    }
    table[n].push_back(dot(cb, b));
    std::vector<unsigned> base_ind = ind;
    unsigned iter = 0;
    while (true)
    {
        iter += 1;
        unsigned max_ind = std::max_element(table[n].begin(), table[n].end() - 1) - table[n].begin();
        if (table[n][max_ind] <= 0.0 && iter <= 100)
        {
            linprog_o result;
            result.success = true;
            result.iter_num = iter;
            for (unsigned i = 0; i < m; i++)
            {
                result.result.push_back(0);
            }
            for (unsigned i = 0; i < n; i++)
            {
                result.result[base_ind[i]] = table[i][m];
            }
            result.obj_value = table[n][m];
            result.base_ind = base_ind;
            return result;
        }
        else if (iter > 100)
        {
            linprog_o result;
            result.success = false;
            result.iter_num = iter;
            result.base_ind = base_ind;
            return result;
        }
        else
        {
            std::vector<double> boundary;
            std::vector<unsigned> non_zero_ind;
            for (unsigned i = 0; i < n; i++)
            {
                if (table[i][max_ind] > 0)
                {
                    boundary.push_back(table[i][m] / table[i][max_ind]);
                    non_zero_ind.push_back(i);
                }
            }
            if (boundary.empty())
            {
                linprog_o result;
                result.success = true;
                result.obj_value = -99999999.0;
                result.iter_num = iter;
                result.base_ind = base_ind;
                std::vector<double> r(m, 0.0);
                result.result = r;
                return result;
            }
            else
            {
                unsigned min_ind = std::min_element(boundary.begin(), boundary.end()) - boundary.begin();
                min_ind = non_zero_ind[min_ind];
                unsigned out = 0;
                for (unsigned i = 0; i < n; i++)
                {
                    if (std::abs(table[min_ind][base_ind[i]] - 1.0) <= 0)
                    {
                        out = i;
                        break;
                    }
                }
                base_ind[out] = max_ind;

                for (unsigned i = 0; i <= n; i++)
                {
                    if (i != min_ind)
                    {
                        double coeff = table[i][max_ind] / table[min_ind][max_ind];
                        for (unsigned j = 0; j <= m; j++)
                        {
                            table[i][j] -= coeff * table[min_ind][j];
                        }
                    }
                }
                double save = table[min_ind][max_ind];
                for (unsigned j = 0; j <= m; j++)
                {
                    table[min_ind][j] /= save;
                }
            }
        }
    }
}

linprog_o mini_optim::simplex(const std::vector<double> &c, const std::vector<std::vector<double>> &A,
                              const std::vector<double> &b)
{
    double M = 999999999.0, zero = 0.00000000001;
    std::vector<std::vector<double>> A_ = A;
    std::vector<double> c_ = c;
    unsigned m = c.size(), n = A.size();
    bool positive = true;
    for (unsigned i = 0; i < n; i++)
    {
        positive = positive && b[i] >= 0;
    }
    if (m <= n || A.empty() || b.empty() || c.empty() || !positive)
    {
        linprog_o result;
        result.success = false;
        result.iter_num = 0;
        return result;
    }
    for (unsigned i = 0; i < n; i++)
    {
        std::vector<double> temp(n, 0);
        temp[i] = 1.0;
        A_[i].insert(A_[i].end(), temp.begin(), temp.end());
    }
    std::vector<double> temp(n, M);
    c_.insert(c_.end(), temp.begin(), temp.end());
    std::vector<unsigned> ind_;
    for (unsigned i = 0; i < n; i++)
    {
        ind_.push_back(m + i);
    }
    auto r = simplex_core(c_, A_, b, ind_);
    auto solution = r.result;
    bool artifi_var_eliminted = true;
    for (unsigned i = 0; i < n; i++)
    {
        artifi_var_eliminted = artifi_var_eliminted && (std::abs(solution[m + i]) <= zero);
    }
    if (r.success && artifi_var_eliminted)
    {
        linprog_o result;
        result.success = true;
        result.iter_num = r.iter_num;
        auto copy = r.result;
        copy.resize(m);
        result.result = copy;
        result.obj_value = r.obj_value;
        result.base_ind = r.base_ind;
        return result;
    }
    else if (r.success && r.result.empty() && artifi_var_eliminted)
    {
        linprog_o result;
        result.success = true;
        result.iter_num = r.iter_num;
        result.obj_value = -99999999.0;
        result.base_ind = r.base_ind;
        return result;
    }
    else
    {
        linprog_o result;
        result.success = false;
        result.iter_num = r.iter_num;
        result.base_ind = r.base_ind;
        return result;
    }
}

linprog_o mini_optim::linprog(const linprog_i &param_input)
{
    unsigned m = param_input.c.size(), n = param_input.Aeq.size();
    if (param_input.c.empty() || (param_input.A.empty() && param_input.Aeq.empty()) || (param_input.b.empty() && param_input.beq.empty()) || m <= n)
    {
        linprog_o result;
        result.success = false;
        result.iter_num = 0;
        return result;
    }
    auto A = param_input.A, Aeq = param_input.Aeq;
    auto b = param_input.b, beq = param_input.beq, c = param_input.c;
    std::vector<std::vector<double>> A_, Aeq_;
    std::vector<double> c_;
    std::vector<unsigned> greater(A.size(), 0);
    for (unsigned i = 0; i < A.size(); i++)
    {
        if (b[i] < 0)
        {
            greater[i] = 1;
            b[i] *= -1;
            for (unsigned j = 0; j < m; j++)
            {
                A[i][j] *= -1;
            }
        }
        std::vector<double> A_row;
        for (unsigned j = 0; j < m; j++)
        {
            A_row.push_back(A[i][j]);
            A_row.push_back(-A[i][j]);
        }
        A_.push_back(A_row);
    }
    for (unsigned i = 0; i < Aeq.size(); i++)
    {
        if (beq[i] < 0)
        {
            beq[i] *= -1;
            for (unsigned j = 0; j < m; j++)
            {
                Aeq[i][j] *= -1;
            }
        }
        std::vector<double> Aeq_row;
        for (unsigned j = 0; j < m; j++)
        {
            Aeq_row.push_back(Aeq[i][j]);
            Aeq_row.push_back(-Aeq[i][j]);
        }
        Aeq_.push_back(Aeq_row);
    }
    for (unsigned i = 0; i < m; i++)
    {
        c_.push_back(c[i]);
        c_.push_back(-c[i]);
    }
    for (unsigned i = 0; i < Aeq_.size(); i++)
    {
        std::vector<double> temp(A_.size(), 0);
        Aeq_[i].insert(Aeq_[i].end(), temp.begin(), temp.end());
    }
    for (unsigned i = 0; i < A_.size(); i++)
    {
        std::vector<double> temp(A_.size(), 0);
        if (greater[i])
        {
            temp[i] = -1.0;
        }
        else
        {
            temp[i] = 1.0;
        }
        A_[i].insert(A_[i].end(), temp.begin(), temp.end());
    }
    std::vector<double> temp(A_.size(), 0);
    c_.insert(c_.end(), temp.begin(), temp.end());
    Aeq_.insert(Aeq_.end(), A_.begin(), A_.end());
    beq.insert(beq.end(), b.begin(), b.end());
    auto r = simplex(c_, Aeq_, beq);
    if (!r.success)
    {
        linprog_o result;
        result.success = false;
        result.iter_num = r.iter_num;
        return result;
    }
    else if (r.result.empty())
    {
        linprog_o result;
        result.success = true;
        result.iter_num = r.iter_num;
        result.obj_value = r.obj_value;
        return result;
    }
    else
    {
        std::vector<double> solution;
        for (unsigned i = 0; i < m; i++)
        {
            solution.push_back(r.result[2 * i] - r.result[2 * i + 1]);
        }
        linprog_o result;
        result.success = true;
        result.iter_num = r.iter_num;
        result.obj_value = r.obj_value;
        result.result = solution;
        return result;
    }
}
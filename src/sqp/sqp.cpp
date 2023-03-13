#include "../../include/mini_optim_bits/sqp.h"
#include <iostream>
using namespace mini_optim;
using namespace std;

sqp_o mini_optim::sqp(const sqp_i &input)
{
    double eta = 0.3, tau = 0.9;
    vector<double> x = input.x0;
    vector<vector<double>> B = eye(input.x0.size());
    vector<double> lambda(input.A.size() + input.Aeq.size(), 0.0);
    unsigned iter = 0;
    while (true)
    {
        cout<<iter<<endl;
        iter++;
        auto A = input.A, Aeq = input.Aeq;
        auto g = grad(input.f, x);
        auto b = add(multiply(A, x), multiply(-1.0, input.b)), beq = add(multiply(Aeq, x), multiply(-1.0, input.beq));
        qp_i in;
        in.H = B;
        in.c = g;
        in.A = multiply(-1.0, A);
        in.b = b;
        in.Aeq = multiply(-1.0, Aeq);
        in.beq = beq;
        auto r = quadprog(in);
        auto p = r.result, lambda_ = r.multiplier, p_lam = add(lambda_, multiply(-1.0, lambda));
        if (dot(p, p) <= input.tol || iter >= input.max_iter)
        {
            sqp_o ret;
            ret.iter = iter;
            ret.success = true;
            ret.result = x;
            ret.obj_value = input.f(x);
            return ret;
        }
        for (auto i = 0; i < b.size(); i++)
        {
            b[i] = min(0.0, b[i]);
            b[i] = fabs(b[i]);
        }
        for (auto i = 0; i < beq.size(); i++)
        {
            beq[i] = fabs(beq[i]);
        }
        double c = accumulate(b.begin(), b.end(), 0.0);
        c = accumulate(beq.begin(), beq.end(), c);
        double mu = 0.0, alpha = 1.0;
        for (auto i = 0; i < r.multiplier.size(); i++)
        {
            r.multiplier[i] = fabs(r.multiplier[i]);
        }
        mu = *max_element(r.multiplier.begin(), r.multiplier.end());
        auto phi = [&](vector<double> x_) -> double
        {
            auto temp1 = add(multiply(input.A, x_), multiply(-1.0, input.b));
            auto temp2 = add(multiply(input.Aeq, x_), multiply(-1.0, input.beq));
            for (auto i = 0; i < temp1.size(); i++)
            {
                temp1[i] = min(0.0, temp1[i]);
                temp1[i] = fabs(temp1[i]);
            }
            for (auto i = 0; i < temp2.size(); i++)
            {
                temp2[i] = min(0.0, temp2[i]);
                temp2[i] = fabs(temp2[i]);
            }
            double c_ = accumulate(temp1.begin(), temp1.end(), 0.0);
            c_ = accumulate(temp2.begin(), temp2.end(), c_);
            return input.f(x_) + mu * c_;
        };

        double eps = 0.001;
        double D = (phi(add(x, multiply(eps, p))) - phi(x)) / eps;
        unsigned exit = 0;
        while (exit <= 50)
        {
            exit++;
            eps *= 0.9;
            double g_ = (phi(add(x, multiply(eps, p))) - phi(x)) / eps;
            if (fabs(D - g_) <= 0.000000001)
            {
                break;
            }
            D = g_;
        }

        exit = 0;
        while (phi(add(x, multiply(alpha, p))) > phi(x) + eta * alpha * D && exit <= 100)
        {
            exit++;
            alpha *= tau;
        }
        auto save = x;
        x = add(x, multiply(alpha, p));
        auto res = add(x, multiply(-1.0, save));
        lambda = add(lambda, multiply(alpha, p_lam));
        auto s = multiply(alpha, p), y = add(grad(input.f, x), multiply(-1.0, g));
        double temp1 = dot(s, multiply(B, s)), temp2 = dot(s, y), theta = 1.0;
        if (temp2 < 0.2 * temp1)
        {
            theta = 0.8 * temp1 / (temp1 - temp2);
        }
        auto temp3 = multiply(B, s);
        auto rk = add(multiply(theta, y), multiply(1.0 - theta, temp3));
        B = add(B, multiply(-1.0 / temp1, multiply(temp3, temp3)));
        B = add(B, multiply(1.0 / dot(s, rk), multiply(rk, rk)));
    }
}

# some solvers for numerical optimization
This lib contains methods of line programming, feasible direction, quadratic programming, SQP and derivative-free algorithm. Using only C++ STL.
This lib can solve medium or small scale problems, or be used for course experiments.


# progress(this lib needs further implementation)
Derivative-free method is under development, and SQP and feasible direction methods do not support non-linear constraints cuurently. Moreover it's expected that more methods like unconstrained function solvers, and the documentation or help reference will be added in the future. 

# An SQP example
> SQP method needs sub-libs of  line programming and quadratic programming
```c++
#include <iostream>
#include "/include/mini_optim.h"
using namespace mini_optim;
using namespace std;
int main(int, char **)
{
    // define a obj_fun
    // see 'using obj_fun = std::function<double(std::vector<double> &)>;' in 'line_search.cpp'
    obj_fun f = [](vector<double> &x) -> double
    {
        return pow(1 - 0.1 * x[0], 2) + 100 * pow(x[1] - 0.5 * x[0] * x[0], 2) + pow(1 - 0.1 * x[2], 2) + 100 * pow(x[3] - 0.5 * x[2] * x[2], 2);
    };

    // define a standard input form for function sqp()
    sqp_i in;
    
    // primary point
    vector<double> x0 = {1, 0.5, 1, 0};
    
    // inequal constaints A*x>=b
    vector<vector<double>> A = {{-1, -1, -1, 0}, {-2, -3, -4, -5}};
    vector<double> b = {-3, -15};

    // equal constaints Aeq*x=beq
    // note: when equal constaints added, some primary points lead to the failure of the iteration. I'm figuring out what has caused the issue.
    // comment out the following 2 lines if it caused issue frequently
    vector<vector<double>> Aeq = {{1, 0, 2, 0}};
    vector<double> beq = {3};
    
    // assign the params
    in.Aeq = Aeq;
    in.beq = beq;
    in.A = A;
    in.b = b;
    in.x0 = x0;
    in.f = f;
    in.max_iter = 500;

    auto r1 = sqp(in);
    
    return 0;
}


```

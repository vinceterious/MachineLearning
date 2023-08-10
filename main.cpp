#include <iostream>
#include "matrix/Matrix.hpp"
#include <fstream>
#include <string>
#include <vector>

namespace 
{
std::vector<std::string> split(std::string line, std::string delim)
{
    std::vector<std::string> result;
    size_t pos = 0;
    while ((pos = line.find(delim)) != std::string::npos)
    {
        result.push_back(line.substr(0, pos));
        line.erase(0, pos + delim.length());
    }
    return result;
}

}

int main()
{
    constexpr std::size_t M1{4899};
    constexpr std::size_t N1{12};
    constexpr std::size_t N2{1};
    using MatFeature = Matrix<double,M1,N1>;
    MatFeature* m = new MatFeature{1};
    using MatF = Matrix<double,M1,N2>;
    MatF* y = new MatF{1};
    using MatTheta = Matrix<double,N1,N2>;
    MatTheta* theta = new MatTheta{0};
    std::ifstream infile("winequality-white.csv");
    std::string line;
    if (infile.is_open()) 
    {
        std::size_t i{0};
        while (std::getline(infile, line))
        {
            line += ';';
            std::vector<std::string> r = split(line,";");
            if(i==0) {i++; continue;}
            for( std::size_t j{0} ; j < r.size() - 1; j++ )
            {
                m->access(i,j) = std::stod(r.at(j));
            }
            y->access(i,0) = std::stod(r.back());
            i++;
        }
    }
    /* Gradient descent */
    MatF modelMinusY = m->multiplication<1>(*theta).soustraction(*y);
    auto res = (1.0/static_cast<double>(2.0*M1))*(modelMinusY.squaredEach().sum());
    double norma = (1.0/static_cast<double>(M1));
    for(int i{0} ; i < 5000 ; i++)
    {
        modelMinusY = m->multiplication<1>(*theta).soustraction(*y);
        auto grad = m->transpose().multiplication<1>(modelMinusY).multiplicationEach(norma);
        *theta = theta->soustraction(grad.multiplicationEach(0.00001));
        //std::cout << "It" << i << " " <<*theta << "\n";
    }

    /* Normal Equation via LU decomposition */
    MatTheta* thetaNormalEq = new MatTheta{1};
    auto A = m->transpose().multiplication<N1>(*m);
    auto b = m->transpose().multiplication<N2>(*y);
    *thetaNormalEq = b.solveLinearEquationViaLU(A.decompositionLU());

    /* Coeficient determination for both algorithm */
    MatF moy {y->moyenne()};
    auto residueV = y->soustraction(moy).squaredEach().sum();
    auto residueUGradDescent = y->soustraction(m->multiplication<1>(*theta)).squaredEach().sum();
    auto residueUNormalEq = y->soustraction(m->multiplication<1>(*thetaNormalEq)).squaredEach().sum();

    /* Print result */
    std::cout << "Regression lineair result:" << std::endl;
    std::cout << "Theta for gradient descent\n" << *theta << std::endl;
    std::cout << "Theta for Normal Equation\n" << *thetaNormalEq << std::endl;

    std::cout << "Performance for Gradient Descent algorithm: R2 factor " <<  1 - residueUGradDescent/residueV << std::endl;
    std::cout << "Performance for Normal Equation algorithm: R2 factor " <<  1 - residueUNormalEq/residueV << std::endl;

    delete[] m,y,theta,thetaNormalEq;
    return 0;
}
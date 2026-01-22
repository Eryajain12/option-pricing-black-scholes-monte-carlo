#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <algorithm>

class Options {
protected:
    double S;        // stock price
    double K;        // strike price
    double T;        // time to maturity
    double r;        // risk-free interest rate
    double sigma;    // implied volatility
    std::string optionType; // call or put

public:
    Options(double S, double K, double T, double r, double sigma, std::string optionType)
        : S(S), K(K), T(T), r(r), sigma(sigma), optionType(optionType) {}

    virtual double price() = 0;
};

// -------------------- Black-Scholes --------------------
class BlackScholes : public Options {
public:
    BlackScholes(double S, double K, double T, double r, double sigma, std::string optionType)
        : Options(S, K, T, r, sigma, optionType) {}

    double d1() {
        return (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    }

    double d2() {
        return d1() - sigma * sqrt(T);
    }

    double cumulativeNormal(double x) {
        const double Pi = 3.141592653589793;
        double L = fabs(x);
        double k = 1.0 / (1.0 + 0.2316419 * L);
        double w = 1.0 - 1.0 / sqrt(2 * Pi) * exp(-L * L / 2.0) *
            (0.31938153 * k
           - 0.356563782 * k * k
           + 1.781477937 * pow(k, 3)
           - 1.821255978 * pow(k, 4)
           + 1.330274429 * pow(k, 5));

        if (x < 0.0)
            w = 1.0 - w;

        return w;
    }

    double price() override {
        double d1_val = d1();
        double d2_val = d2();

        if (optionType == "call") {
            return S * cumulativeNormal(d1_val)
                 - K * exp(-r * T) * cumulativeNormal(d2_val);
        }
        else if (optionType == "put") {
            return K * exp(-r * T) * cumulativeNormal(-d2_val)
                 - S * cumulativeNormal(-d1_val);
        }

        return 0.0;
    }
};

// -------------------- Monte Carlo --------------------
class MonteCarlo : public Options {
private:
    int numSimulations;

public:
    MonteCarlo(double S, double K, double T, double r, double sigma,
               int numSimulations, std::string optionType)
        : Options(S, K, T, r, sigma, optionType),
          numSimulations(numSimulations) {}

    double price() override {
        double payoffSum = 0.0;
        double dt = T;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 1);

        for (int i = 0; i < numSimulations; i++) {
            double ST = S * exp((r - 0.5 * sigma * sigma) * dt
                         + sigma * sqrt(dt) * d(gen));

            double payoff = 0.0;

            if (optionType == "call")
                payoff = std::max(ST - K, 0.0);
            else if (optionType == "put")
                payoff = std::max(K - ST, 0.0);

            payoffSum += payoff;
        }

        return exp(-r * T) * (payoffSum / numSimulations);
    }
};

// -------------------- Main --------------------
int main() {
    double S, K, T, r, sigma;
    int numSimulations;
    std::string optionType, modelChoice;

    std::cout << "Enter the current asset price (S): ";
    std::cin >> S;

    std::cout << "Enter the strike price (K): ";
    std::cin >> K;

    std::cout << "Enter the time to maturity in years (T): ";
    std::cin >> T;

    std::cout << "Enter the risk-free rate (r) as a decimal: ";
    std::cin >> r;

    std::cout << "Enter the implied volatility (sigma) as a decimal: ";
    std::cin >> sigma;

    std::cout << "Enter the option type ('call' or 'put'): ";
    std::cin >> optionType;

    std::cout << "Which pricing model would you like to use? ('Black-Scholes' or 'MonteCarlo'): ";
    std::cin >> modelChoice;

    if (modelChoice == "MonteCarlo") {
        std::cout << "Enter the number of simulations: ";
        std::cin >> numSimulations;

        MonteCarlo option(S, K, T, r, sigma, numSimulations, optionType);
        double optionPrice = option.price();

        std::cout << "Monte Carlo " << optionType
                  << " Option Price: " << optionPrice << std::endl;
    }
    else if (modelChoice == "Black-Scholes") {
        BlackScholes option(S, K, T, r, sigma, optionType);
        double optionPrice = option.price();

        std::cout << "Black-Scholes " << optionType
                  << " Option Price: " << optionPrice << std::endl;
    }
    else {
        std::cout << "Invalid model choice!" << std::endl;
    }

    return 0;
    
}
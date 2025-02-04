#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Neuronio {
protected:
    vector<double> pesos;
    double bias;

public:
    Neuronio(const vector<double>& pesos, double bias)
        : pesos(pesos), bias(bias) {}

    virtual double predict(const vector<double>& entradas) const = 0;
    virtual ~Neuronio() {}
};

class NeuronioReLU : public Neuronio {
public:
    NeuronioReLU(const vector<double>& pesos, double bias)
        : Neuronio(pesos, bias) {}

    double predict(const vector<double>& entradas) const override {
        if (entradas.size() != pesos.size()) {
            cerr << "Erro: Número de entradas não corresponde ao número de pesos!" << endl;
            return 0.0;
        }

        double soma = bias;
        for (size_t i = 0; i < entradas.size(); ++i) {
            soma += entradas[i] * pesos[i];
        }

        return max(0.0, soma);
    }
};

int main() {
    vector<Neuronio*> neuronios;
    neuronios.push_back(new NeuronioReLU({0.2, 0.4}, -1.5));  // Caso 1
    neuronios.push_back(new NeuronioReLU({0.2, 0.4}, -0.5));  // Caso 2

    vector<vector<double>> entradas = {
        {0.3, 2.0},  // Caso 1
        {0.3, 2.0},  // Caso 2
    };

    for (size_t i = 0; i < neuronios.size(); ++i) {
        cout << "Saída do Neurônio " << i + 1 << " (ReLU): "
             << neuronios[i]->predict(entradas[i]) << endl;
    }
    for (auto neur : neuronios) {
        delete neur;
    }
    return 0;
}

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
using namespace std;

double get_energy(vector<vector<int>> lattice);
pair<vector<double>, vector<double>> metropolis(vector<int> lattice, int times, double bj, double energy);
vector<vector<int>> lattice(int N);
vector<vector<int>> convolve(vector<vector<int>> lattice);
double sum(vector<vector<int>> lattice);
vector<vector<int>> multiply(vector<vector<int>> lattice, vector<vector<int>> conv);

//N x N grid
    const int N = 50;

int main(){
    
    auto lattice_p = lattice(N);
    for(size_t i = 0; i < lattice_p.size(); i++){
        for(size_t j = 0; j < lattice_p[i].size(); i++){
            lattice_p[i][j] > 0.75 ? lattice_p[i][j] = -1 : lattice_p[i][j] = 1;
        }
    }
    
    auto lattice_n = lattice(N);
    for(size_t i = 0; i < lattice_n.size(); i++){
        for(size_t j = 0; j < lattice_n[i].size(); i++){
            lattice_n[i][j] > 0.75 ? lattice_n[i][j] = 1 : lattice_n[i][j] = -1;
        }
    }
    
    auto result = metropolis(lattice_p, 1000000, 0.7, get_energy(lattice_p));
    vector<double> spins = result.first;
    vector<double> energies = result.second;

    for(double i : spins){
        cout << i <<endl;
    }

}

vector<vector<int>> lattice(int N){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(0.0, 1.0);
    vector<vector<int>> v;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            v[i][j] = abs(dist(gen));
        }
    }
    return v;

}

double sum(vector<vector<int>> lattice){
    double sum = 0;
    for(auto row : lattice){
        sum += accumulate(row.begin(), row.end(), 0.0);
    }
    return sum;
}

vector<vector<int>> multiply(vector<vector<int>> lattice, vector<vector<int>> conv){
    vector<vector<int>> m;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                m[i][j] += lattice[i][k] * conv[k][j];
            }
        }
    }
    return m;

}

double get_energy(vector<vector<int>> lattice){
    auto energy = multiply(lattice, convolve(lattice));
    return -sum(energy);
}

pair<vector<double>, vector<double>> metropolis(vector<vector<int>> lattice, int times, double bj, double energy){
    vector<double> spins;
    vector<double> energies;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(0.0, 1.0);
    for(int i = 0; i <= times; i++){
        int x = abs(dist(gen));
        int y = abs(dist(gen));
        int spin_i = lattice[x][y];
        int spin_f = -spin_i;
        double E_i = 0;
        double E_f = 0;
        
        if (x>0){
            E_i += -spin_i*lattice[x-1][y];
            E_f += -spin_f*lattice[x-1][y];
        }
        if (x<N-1){
            E_i += -spin_i*lattice[x+1][y];
            E_f += -spin_f*lattice[x+1][y];
        }
        if (y>0){
            E_i += -spin_i*lattice[x][y-1];
            E_f += -spin_f*lattice[x][y-1];
        }
        if (y<N-1){
            E_i += -spin_i*lattice[x][y+1];
            E_f += -spin_f*lattice[x][y+1];
        }
        double dE = E_f - E_i;
        
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dist(0.0, 1.0);
        if(dE > 0 && (abs(dist(gen)) < exp(-bj*dE))){
            lattice[x][y] = spin_f;
            energy += dE;
        }
        else if(dE < 0){
            lattice[x][y] = spin_f;
            energy += dE;
        }

        spins.push_back(sum(lattice));
        energies.push_back(energy);
    }
    return make_pair(spins, energies);



}

vector<vector<int>> convolve(vector<vector<int>> lattice){
    bool kern[3][3] = {{false, true, false},
                 {true, false, true},
                 {false, true, false}};
    vector<vector<int>> conv;
    for(size_t i = 0; i < lattice.size(); i++){
        for(size_t j = 0; j < lattice.size(); j++){
            int sum = 0;
            for(int n = 0; n<3; n++){
                for(int m = 0; m<3; m++){
                    if(i+n-1 < 0 || j+m-1 < 0){
                        sum += 0;
                    }
                    if(i+n-1 >= N || j+m-1 >= N){
                        sum += 0;
                    }
                    else{
                        sum += lattice[i+n-1][j+m-1]*kern[m][n];
                    }
            }   
            }
            conv[i][j] = sum;
    }   
    }
    return conv;
}
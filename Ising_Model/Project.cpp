#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <fstream>
#include <filesystem>
using namespace std;

double get_energy(vector<vector<int>> lattice);
pair<vector<double>, vector<double>> metropolis(vector<vector<int>> lattice, int times, double bj, double energy);
vector<vector<double>> make_lattice(int N);
vector<vector<int>> convolve(vector<vector<int>> lattice);
double total_sum(const vector<vector<int>>& lattice);

//N x N grid
    const int N = 50;

int main(){

    const int times = 10000;
    
    auto lattice_po = make_lattice(N);
    vector<vector<int>> lattice_p(N, vector<int>(N, 0));
    for(size_t i = 0; i < lattice_po.size(); i++){
        for(size_t j = 0; j < lattice_po[i].size(); j++){
            lattice_po[i][j] > 0.75 ? lattice_p[i][j] = -1 : lattice_p[i][j] = 1;
        }
    }
    
    auto lattice_no = make_lattice(N);
    vector<vector<int>> lattice_n(N, vector<int>(N, 0));
    for(size_t i = 0; i < lattice_no.size(); i++){
        for(size_t j = 0; j < lattice_no[i].size(); j++){
            lattice_no[i][j] > 0.75 ? lattice_n[i][j] = 1 : lattice_n[i][j] = -1;
        }
    }
    
    auto result = metropolis(lattice_p, times, 0.2, get_energy(lattice_p));
    vector<double> spinsp = result.first;
    vector<double> energiesp = result.second;
    
    ofstream filep("C:\\Users\\Jason\\Desktop\\Study Material\\PH1050\\Project\\datap.txt");
    double count = 0;
    for(double i : spinsp){
        filep << i/N*N << ' ' << energiesp[count] << ' ' << count << endl;
        count++;
    }
    filep.close();
    
    result = metropolis(lattice_n, times, 0.2, get_energy(lattice_n));
    vector<double> spinsn = result.first;
    vector<double> energiesn = result.second;
    
    ofstream filen("C:\\Users\\Jason\\Desktop\\Study Material\\PH1050\\Project\\datan.txt");
    count = 0;
    for(double i : spinsn){
        filen << i/N*N << ' ' << energiesn[count] << ' ' << count << endl;
        count++;
    }
    filen.close();
    
    ofstream filebj("C:\\Users\\Jason\\Desktop\\Study Material\\PH1050\\Project\\databj.txt");
    vector<double> bjs;
    for(double bj = 0.1; bj<=2; bj+=0.05){
        auto metrop = metropolis(lattice_p, times, bj, get_energy(lattice_p));
        vector<double> spinsp = metrop.first;
        double avgSpinp = accumulate(spinsp.begin(), spinsp.end(), 0.0) / (spinsp.size()*N*N);
        auto metron = metropolis(lattice_n, times, bj, get_energy(lattice_n));
        vector<double> spinsn = metron.first;
        double avgSpinn = accumulate(spinsn.begin(), spinsn.end(), 0.0) / (spinsn.size()*N*N);
        filebj << bj << ' ' << avgSpinp << ' ' << avgSpinn << endl;
    }
    filebj.close();

}

vector<vector<double>> make_lattice(int N){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(0.0, 1.0);
    vector<vector<double>> v(N, vector<double>(N,0));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            v[i][j] = dist(gen);
        }
    }
    return v;

}

double total_sum(const vector<vector<int>>& lattice){
    double sum = 0;
    for(auto row : lattice){
        sum += accumulate(row.begin(), row.end(), 0.0);
    }
    return sum;
}


double get_energy(vector<vector<int>> lattice){
    auto conv = convolve(lattice);
    double e = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            e += lattice[i][j] * conv[i][j];
    return -e;
}

pair<vector<double>, vector<double>> metropolis(vector<vector<int>> lattice, int times, double bj, double energy){
    vector<double> spins;
    vector<double> energies;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist_int(0, N-1);
    uniform_real_distribution<> prob(0.0, 1.0);
    for(int i = 0; i < times; i++){
        int x = dist_int(gen);
        int y = dist_int(gen);
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
        
        
        if(dE > 0 && (prob(gen) < exp(-bj*dE))){
            lattice[x][y] = spin_f;
            energy += dE;
        }
        else if(dE < 0){
            lattice[x][y] = spin_f;
            energy += dE;
        }

        spins.push_back(total_sum(lattice));
        energies.push_back(energy);
    }
    return make_pair(spins, energies);



}

vector<vector<int>> convolve(vector<vector<int>> lattice){
    int kern[3][3] = {{0, 1, 0},
                      {1, 0, 1},
                      {0, 1, 0}};
    vector<vector<int>> conv(N, vector<int>(N, 0));
    for(size_t i = 0; i < lattice.size(); i++){
        for(size_t j = 0; j < lattice.size(); j++){
            int sum = 0;
            for(int n = 0; n<3; n++){
                for(int m = 0; m<3; m++){
                    if(i+n-1 < 0 || j+m-1 < 0 || i+n-1 >= N || j+m-1 >= N){
                        sum += 0;
                    }
                    else{
                        sum += lattice[i+n-1][j+m-1]*kern[n][m];
                    }
            }   
            }
            conv[i][j] = sum;
    }   
    }
    return conv;
}
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <TCanvas.h>
#include <TGraph.h>
#include <TAxis.h>
#include <TStyle.h>

using namespace std;

// --- Function declarations ---
double get_energy(vector<vector<int>> lattice);
pair<vector<double>, vector<double>> metropolis(vector<vector<int>> lattice, int times, double bj, double energy);
vector<vector<double>> make_lattice(int N);
vector<vector<int>> convolve(vector<vector<int>> lattice);
double total_sum(const vector<vector<int>>& lattice);

// --- Constants ---
const int N = 50;

// --- Main program ---
int main() {

    const int times = 10000;

    // Initial lattice with mostly +1 spins
    auto lattice_po = make_lattice(N);
    vector<vector<int>> lattice_p(N, vector<int>(N, 0));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            lattice_po[i][j] > 0.75 ? lattice_p[i][j] = -1 : lattice_p[i][j] = 1;

    // Initial lattice with mostly -1 spins
    auto lattice_no = make_lattice(N);
    vector<vector<int>> lattice_n(N, vector<int>(N, 0));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            lattice_no[i][j] > 0.75 ? lattice_n[i][j] = 1 : lattice_n[i][j] = -1;

    // --- Create canvases and graphs ---
    TCanvas *c1 = new TCanvas("c1", "Ising Monte Carlo", 1200, 900);
    c1->Divide(2, 2);

    TGraph *grps = new TGraph(); // Spin vs time (positive init)
    grps->SetLineColor(kRed);

    TGraph *grpe = new TGraph(); // Energy vs time (positive init)
    grpe->SetLineColor(kRed);

    TGraph *grns = new TGraph(); // Spin vs time (negative init)
    grns->SetLineColor(kBlack);

    TGraph *grne = new TGraph(); // Energy vs time (negative init)
    grne->SetLineColor(kBlack);

    // --- Run simulations ---
    auto resultp = metropolis(lattice_p, times, 0.2, get_energy(lattice_p));
    auto resultn = metropolis(lattice_n, times, 0.2, get_energy(lattice_n));

    vector<double> spinsp = resultp.first;
    vector<double> energiesp = resultp.second;
    vector<double> spinsn = resultn.first;
    vector<double> energiesn = resultn.second;

    // --- Fill graphs directly ---
    for (int i = 0; i < times; i++) {
        double t = (double)i / times;
        grps->SetPoint(i, t, spinsp[i] / (N * N));
        grpe->SetPoint(i, t, energiesp[i]);
        grns->SetPoint(i, t, spinsn[i] / (N * N));
        grne->SetPoint(i, t, energiesn[i]);
    }

    // --- Draw first canvas ---
    c1->cd(1);
    grps->SetTitle("Positive Init: Average Spin;Time;#bar{m}");
    grps->Draw("AL");

    c1->cd(2);
    grpe->SetTitle("Positive Init: Energy;Time;E/J");
    grpe->Draw("AL");

    c1->cd(3);
    grns->SetTitle("Negative Init: Average Spin;Time;#bar{m}");
    grns->Draw("AL");

    c1->cd(4);
    grne->SetTitle("Negative Init: Energy;Time;E/J");
    grne->Draw("AL");

    c1->Update();

    // --- Second canvas: Temperature dependence ---
    TCanvas *c2 = new TCanvas("c2", "Temperature Dependence", 800, 600);
    TGraph *grbjp = new TGraph();
    TGraph *grbjn = new TGraph();

    grbjp->SetLineColor(kRed);
    grbjp->SetMarkerColor(kRed);
    grbjp->SetMarkerStyle(20);

    grbjn->SetLineColor(kBlack);
    grbjn->SetMarkerColor(kBlack);
    grbjn->SetMarkerStyle(20);

    int idx = 0;
    for (double bj = 0.1; bj <= 2.0; bj += 0.05) {
        auto metrop = metropolis(lattice_p, times, bj, get_energy(lattice_p));
        double avgSpinp = accumulate(metrop.first.begin(), metrop.first.end(), 0.0) / (metrop.first.size() * N * N);

        auto metron = metropolis(lattice_n, times, bj, get_energy(lattice_n));
        double avgSpinn = accumulate(metron.first.begin(), metron.first.end(), 0.0) / (metron.first.size() * N * N);

        grbjp->SetPoint(idx, bj, avgSpinp);
        grbjn->SetPoint(idx, bj, avgSpinn);
        idx++;
    }

    grbjp->SetTitle("Magnetization vs Temperature;Temperature (1/#beta J);#bar{m}");
    grbjp->Draw("ALP");
    grbjn->Draw("LP SAME");

    c2->Update();

    cout << "Simulation complete! Close the ROOT windows to end." << endl;
    return 0;
}

// --- Helper functions ---

vector<vector<double>> make_lattice(int N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(0.0, 1.0);
    vector<vector<double>> v(N, vector<double>(N, 0));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            v[i][j] = dist(gen);
    return v;
}

double total_sum(const vector<vector<int>> &lattice) {
    double sum = 0;
    for (auto &row : lattice)
        sum += accumulate(row.begin(), row.end(), 0.0);
    return sum;
}

vector<vector<int>> convolve(vector<vector<int>> lattice) {
    int kern[3][3] = {{0, 1, 0},
                      {1, 0, 1},
                      {0, 1, 0}};
    vector<vector<int>> conv(N, vector<int>(N, 0));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int n = 0; n < 3; n++)
                for (int m = 0; m < 3; m++) {
                    int ni = i + n - 1;
                    int mj = j + m - 1;
                    if (ni >= 0 && mj >= 0 && ni < N && mj < N)
                        sum += lattice[ni][mj] * kern[n][m];
                }
            conv[i][j] = sum;
        }
    return conv;
}

double get_energy(vector<vector<int>> lattice) {
    auto conv = convolve(lattice);
    double e = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            e += lattice[i][j] * conv[i][j];
    return -e;
}

pair<vector<double>, vector<double>> metropolis(vector<vector<int>> lattice, int times, double bj, double energy) {
    vector<double> spins;
    vector<double> energies;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist_int(0, N - 1);
    uniform_real_distribution<> prob(0.0, 1.0);

    for (int i = 0; i < times; i++) {
        int x = dist_int(gen);
        int y = dist_int(gen);
        int spin_i = lattice[x][y];
        int spin_f = -spin_i;

        double E_i = 0, E_f = 0;
        if (x > 0) {
            E_i += -spin_i * lattice[x - 1][y];
            E_f += -spin_f * lattice[x - 1][y];
        }
        if (x < N - 1) {
            E_i += -spin_i * lattice[x + 1][y];
            E_f += -spin_f * lattice[x + 1][y];
        }
        if (y > 0) {
            E_i += -spin_i * lattice[x][y - 1];
            E_f += -spin_f * lattice[x][y - 1];
        }
        if (y < N - 1) {
            E_i += -spin_i * lattice[x][y + 1];
            E_f += -spin_f * lattice[x][y + 1];
        }

        double dE = E_f - E_i;

        if (dE <= 0 || prob(gen) < exp(-bj * dE)) {
            lattice[x][y] = spin_f;
            energy += dE;
        }

        spins.push_back(total_sum(lattice));
        energies.push_back(energy);
    }
    return make_pair(spins, energies);
}

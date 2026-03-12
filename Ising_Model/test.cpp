#include <TCanvas.h>
#include <TGraph.h>
#include <TAxis.h>
#include <TStyle.h>

void four_graphs() {
    // Create a canvas and divide it into 2x2 pads
    TCanvas *c1 = new TCanvas("c1", "4 Subplots Example", 900, 700);
    c1->Divide(2, 2);

    // Create 4 example graphs
    const int N = 100;
    double x[N], y1[N], y2[N], y3[N], y4[N];

    for (int i = 0; i < N; i++) {
        x[i] = i * 0.1;
        y1[i] = sin(x[i]);
        y2[i] = cos(x[i]);
        y3[i] = exp(-0.05 * i) * sin(2 * x[i]);
        y4[i] = log(1 + i * 0.1);
    }

    TGraph *g1 = new TGraph(N, x, y1);
    TGraph *g2 = new TGraph(N, x, y2);
    TGraph *g3 = new TGraph(N, x, y3);
    TGraph *g4 = new TGraph(N, x, y4);

    // Style settings (optional)
    g1->SetLineColor(kRed);
    g2->SetLineColor(kBlue);
    g3->SetLineColor(kGreen+2);
    g4->SetLineColor(kMagenta);

    // Draw each graph in its pad
    c1->cd(1);
    g1->SetTitle("Graph 1: sin(x)");
    g1->Draw("AL");

    c1->cd(2);
    g2->SetTitle("Graph 2: cos(x)");
    g2->Draw("AL");

    c1->cd(3);
    g3->SetTitle("Graph 3: damped sine");
    g3->Draw("AL");

    c1->cd(4);
    g4->SetTitle("Graph 4: log(1+x)");
    g4->Draw("AL");

    c1->Update();
}

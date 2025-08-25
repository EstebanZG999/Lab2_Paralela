// ============================================
// Parte 2
// Suma de arreglo con OpenMP: reduction vs atomic vs critical
// Imprime tiempos y speedup en CSV y valida exactitud.
// Uso:
//   g++ -fopenmp -o lab2_part2 lab2_part2.cpp
//   ./lab2_part2 [N] [chunk]
//   ./lab2_part2            # N=10'000'000, chunk=1024
//   ./lab2_part2 20000000 256
// ============================================

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

int main(int argc, char** argv) {
    size_t N = 10'000'000;   // tamaño
    int chunk = 1024;        // chunk 

    if (argc >= 2) N = strtoull(argv[1], nullptr, 10);
    if (argc >= 3) chunk = max(1, atoi(argv[2])); 

    vector<double> a(N);

    // Inicialización determinista
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)N; ++i) {
        a[i] = 0.5 + 1e-6 * (i % 1000);
    }

    // --- Baseline secuencial ---
    double t0 = omp_get_wtime();
    double seq_sum = 0.0;
    for (size_t i = 0; i < N; ++i) seq_sum += a[i];
    double t1 = omp_get_wtime();
    double seq_time = t1 - t0;

    vector<int> thread_configs = {1, 2, 4, 8};

    // CSV header
    cout << "variant,threads,N,chunk,time_seconds,speedup,abs_error\n";

    auto print_row = [&](const string& variant, int T, double time, double sum){
        double speed = seq_time / max(time, 1e-12);
        double err = fabs(sum - seq_sum);
        cout << variant << "," << T << "," << N << "," << chunk << ","
             << fixed << setprecision(6) << time << ","
             << fixed << setprecision(2) << speed << ","
             << scientific << setprecision(3) << err << "\n";
    };

    for (int T : thread_configs) {
        omp_set_num_threads(T);

        // REDUCTION
        double red_sum = 0.0;
        double r0 = omp_get_wtime();
        #pragma omp parallel for reduction(+:red_sum) schedule(static, chunk)
        for (long long i = 0; i < (long long)N; ++i) {
            red_sum += a[i];
        }
        double r1 = omp_get_wtime();
        print_row("reduction", T, r1 - r0, red_sum);

        // ATOMIC
        double atomic_sum = 0.0;
        double a0 = omp_get_wtime();
        #pragma omp parallel for schedule(static, chunk)
        for (long long i = 0; i < (long long)N; ++i) {
            double v = a[i];
            #pragma omp atomic
            atomic_sum += v;
        }
        double a1 = omp_get_wtime();
        print_row("atomic", T, a1 - a0, atomic_sum);

        // CRITICAL
        double crit_sum = 0.0;
        double c0 = omp_get_wtime();
        #pragma omp parallel for schedule(static, chunk)
        for (long long i = 0; i < (long long)N; ++i) {
            double v = a[i];
            #pragma omp critical
            {
                crit_sum += v;
            }
        }
        double c1 = omp_get_wtime();
        print_row("critical", T, c1 - c0, crit_sum);
    }

    cerr << "Secuencial: " << fixed << setprecision(6) << seq_time << " s"
         << " | N=" << N << " | chunk=" << chunk << "\n";
    cerr << "Máx. hilos OpenMP: " << omp_get_max_threads() << "\n";
    return 0;
}

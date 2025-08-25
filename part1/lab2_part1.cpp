// ================================
// Parte 1
// Problema N-elementos: sqrt sobre arreglo grande
// Permite variar scheduler (static|dynamic|guided) y chunk
// Imprime tiempos (seq y omp) y speedup en CSV
// ================================

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

enum class Sched { STATIC, DYNAMIC, GUIDED };

Sched parse_sched(const string& s) {
    string t = s;
    for (auto &c : t) c = tolower(c);
    if (t == "static")  return Sched::STATIC;
    if (t == "dynamic") return Sched::DYNAMIC;
    if (t == "guided")  return Sched::GUIDED;
    cerr << "Aviso: scheduler desconocido '" << s << "', usando 'static'.\n";
    return Sched::STATIC;
}

const char* sched_name(Sched s) {
    switch (s) {
        case Sched::STATIC:  return "static";
        case Sched::DYNAMIC: return "dynamic";
        case Sched::GUIDED:  return "guided";
    }
    return "static";
}

int main(int argc, char** argv) {
    // --------- Parametrización por CLI ----------
    // Uso:
    //   ./lab2_part1 [N] [scheduler] [chunk]
    // Ejemplos:
    //   ./lab2_part1
    //   ./lab2_part1 20000000 static 1024
    //
    // N: tamaño del arreglo (default 10'000'000)
    // scheduler: static | dynamic | guided (default static)
    // chunk: tamaño de bloque (default 1024)

    size_t N = 10'000'000;      // > 100,000
    Sched sched = Sched::STATIC;
    int chunk = 1024;

    if (argc >= 2) N = strtoull(argv[1], nullptr, 10);
    if (argc >= 3) sched = parse_sched(argv[2]);
    if (argc >= 4) chunk = max(1, atoi(argv[3]));

    // --------- Preparación de datos ----------
    vector<double> a(N), out_seq(N), out_omp(N);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)N; ++i) {
        // patrón determinista
        a[i] = 0.5 + 0.000001 * (i % 1000);
    }

    // --------- Secuencial ----------
    double t0 = omp_get_wtime();
    volatile double checksum_seq = 0.0; // volatile para que no eliminen el cálculo
    for (size_t i = 0; i < N; ++i) {
        double v = sqrt(a[i]);
        out_seq[i] = v;
        checksum_seq += v;
    }
    double t1 = omp_get_wtime();
    double seq_time = t1 - t0;

    // Campos: N,scheduler,chunk,threads,time_seconds,speedup,checksum_match
    cout << "N,scheduler,chunk,threads,time_seconds,speedup,checksum_match\n";

    // --------- Pruebas con hilos {1,2,4,8} ----------
    vector<int> thread_configs = {1, 2, 4, 8};
    for (int T : thread_configs) {
        omp_set_num_threads(T);

        // Selección del schedule en tiempo de compilación con pragma + runtime
        // Usamos pragma con 'schedule(runtime)' y seteamos con omp_set_schedule.
        switch (sched) {
            case Sched::STATIC:
                omp_set_schedule(omp_sched_static, chunk);
                break;
            case Sched::DYNAMIC:
                omp_set_schedule(omp_sched_dynamic, chunk);
                break;
            case Sched::GUIDED:
                omp_set_schedule(omp_sched_guided, chunk);
                break;
        }

        double t2 = omp_get_wtime();
        volatile double checksum_omp = 0.0;
        #pragma omp parallel
        {
            // Aplicar la planificación elegida
            #pragma omp for schedule(runtime)
            for (long long i = 0; i < (long long)N; ++i) {
                double v = sqrt(a[i]);
                out_omp[i] = v;
            }

            double local_sum = 0.0;
            #pragma omp for schedule(static) nowait
            for (long long i = 0; i < (long long)N; ++i) {
                local_sum += out_omp[i];
            }
            #pragma omp atomic
            checksum_omp += local_sum;
        }
        double t3 = omp_get_wtime();
        double omp_time = t3 - t2;
        double speedup = seq_time / max(omp_time, 1e-12);

        // Verificación de exactitud básica 
        bool checksum_ok = fabs((double)checksum_omp - (double)checksum_seq) < 1e-6 * checksum_seq;

        cout << N << ","
             << sched_name(sched) << ","
             << chunk << ","
             << T << ","
             << fixed << setprecision(6) << omp_time << ","
             << fixed << setprecision(2) << speedup << ","
             << (checksum_ok ? "true" : "false")
             << "\n";
    }

    cerr << "Tiempo secuencial: " << fixed << setprecision(6) << seq_time << " s\n";
    cerr << "Scheduler=" << sched_name(sched) << " | chunk=" << chunk << " | N=" << N << "\n";
    cerr << "Max threads disponibles (OpenMP): " << omp_get_max_threads() << "\n";

    return 0;
}

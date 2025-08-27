// ============================================================
// Parte 3
// Producer–Consumer con buffer acotado 
// - Un productor/consumidor (1–1) y múltiple P–C
// - Protección con mutex + condition_variabl
// - Métricas: tiempo, throughput, esperas (lleno/vacío), tamaño máx. de cola
//
// Compilar:
//   g++ -pthread -o lab2_part3 lab2_part3.cpp
//
// Uso:
//   ./lab2_part3 [N] [capacidad] [P] [C] [work_us]
//   N: total de ítems a producir/consumir
//   capacidad: tamaño del buffer
//   P: #productores
//   C: #consumidores
//   work_us: microsegundos de trabajo simulado por ítem
// ============================================================

#include <bits/stdc++.h>
using namespace std;

struct BoundedBuffer {
    explicit BoundedBuffer(size_t cap)
        : cap_(max<size_t>(1, cap)), buf_(cap_), head_(0), tail_(0), size_(0),
          wait_full_(0), wait_empty_(0), max_size_(0) {}

    void push(int x) {
        unique_lock<mutex> lk(m_);
        // Esperar si está lleno
        while (size_ == cap_) {
            ++wait_full_;
            not_full_.wait(lk);
        }
        buf_[tail_] = x;
        tail_ = (tail_ + 1) % cap_;
        ++size_;
        max_size_ = max(max_size_, size_);
        lk.unlock();
        not_empty_.notify_one();
    }

    int pop() {
        unique_lock<mutex> lk(m_);
        // Esperar si está vacío
        while (size_ == 0) {
            ++wait_empty_;
            not_empty_.wait(lk);
        }
        int x = buf_[head_];
        head_ = (head_ + 1) % cap_;
        --size_;
        lk.unlock();
        not_full_.notify_one();
        return x;
    }

    // Solo para métricas
    size_t waits_full()  const { return wait_full_; }
    size_t waits_empty() const { return wait_empty_; }
    size_t max_size()    const { return max_size_; }
    size_t capacity()    const { return cap_; }

private:
    const size_t cap_;
    vector<int> buf_;
    size_t head_, tail_, size_;
    mutable mutex m_;
    condition_variable not_full_, not_empty_;

    // métricas
    size_t wait_full_;
    size_t wait_empty_;
    size_t max_size_;
};


inline void do_work_us(int work_us) {
    if (work_us <= 0) return;
    // Espera activa corta
    auto t0 = chrono::high_resolution_clock::now();
    while (chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - t0).count() < work_us) {
        asm volatile("" ::: "memory");
    }
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // ----------------- Parámetros -----------------
    size_t N = 1'000'000;         // total de ítems
    size_t capacity = 1024;       // tamaño del buffer
    int P = 1;                    // # productores
    int C = 1;                    // # consumidores
    int work_us = 0;              // trabajo simulado por ítem

    if (argc >= 2) N = strtoull(argv[1], nullptr, 10);
    if (argc >= 3) capacity = strtoull(argv[2], nullptr, 10);
    if (argc >= 4) P = max(1, atoi(argv[3]));
    if (argc >= 5) C = max(1, atoi(argv[4]));
    if (argc >= 6) work_us = max(0, atoi(argv[5]));

    BoundedBuffer buffer(capacity);

    // Contadores atómicos compartidos
    atomic<size_t> produced{0};
    atomic<size_t> consumed{0};

    // Para checksum/validación simple
    const size_t target = N;

    // ----------------- Productores -----------------
    auto producer = [&]() {
        for (;;) {
            size_t id = produced.fetch_add(1, memory_order_relaxed);
            if (id >= target) break;
            int item = static_cast<int>(id); // contenido
            buffer.push(item);
        }
    };

    // ----------------- Consumidores -----------------
    atomic<unsigned long long> checksum{0};
    auto consumer = [&]() {
        for (;;) {
            size_t done = consumed.load(memory_order_relaxed);
            if (done >= target) break;
            int item = buffer.pop();
            do_work_us(work_us);
            checksum.fetch_add(static_cast<unsigned long long>(item), memory_order_relaxed);
            consumed.fetch_add(1, memory_order_relaxed);
            if (consumed.load(memory_order_relaxed) >= target) break;
        }
    };

    // ----------------- Lanzar hilos -----------------
    vector<thread> producers, consumers;
    producers.reserve(P);
    consumers.reserve(C);

    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < P; ++i) producers.emplace_back(producer);
    for (int j = 0; j < C; ++j) consumers.emplace_back(consumer);

    for (auto& th : producers) th.join();


    for (int j = 0; j < C; ++j) {
        buffer.push(-1); // marcador de terminación
    }

    for (auto& th : consumers) th.join();
    auto t1 = chrono::high_resolution_clock::now();

    double secs = chrono::duration<double>(t1 - t0).count();
    double throughput = (target / max(secs, 1e-12));

    // ----------------- Validación -----------------
    bool ok_count = (produced.load() >= target) && (consumed.load() >= target);

    // ----------------- Salida CSV -----------------
    // Columns:
    // N,capacity,P,C,work_us,time_seconds,throughput_items_per_sec,waits_full,waits_empty,max_queue_size,ok_count,checksum
    cout.setf(std::ios::fixed); cout << setprecision(6);
    cout << "N,capacity,P,C,work_us,time_seconds,throughput_items_per_sec,waits_full,waits_empty,max_queue_size,ok_count,checksum\n";
    cout << target << "," << buffer.capacity() << "," << P << "," << C << ","
         << work_us << "," << secs << "," << throughput << ","
         << buffer.waits_full() << "," << buffer.waits_empty() << "," << buffer.max_size() << ","
         << (ok_count ? "true" : "false") << "," << checksum.load() << "\n";


    cerr << "Producer-Consumer terminado | N=" << target
         << " | cap=" << buffer.capacity()
         << " | P=" << P << " | C=" << C
         << " | work_us=" << work_us << "us\n";
    cerr << "Tiempo: " << secs << " s | Throughput: " << throughput << " items/s\n";
    cerr << "Espera por lleno: " << buffer.waits_full()
         << " | Espera por vacío: " << buffer.waits_empty()
         << " | Tamaño máximo de cola: " << buffer.max_size() << "\n";

    return 0;
}

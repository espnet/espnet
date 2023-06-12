#include <threads.h>
int main() {
    mtx_t mutex;
    mtx_init(&mutex, mtx_plain);
}


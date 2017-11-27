#include <cuda_runtime.h>
#include <stdio.h>

#define HANDLE_ERROR(err) {                                                    \
    if ((err) != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %d\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    }}

#define IX(i, j) ((i) + (N + 2) * (j))
#define SWAP(x0, x)                                                            \
    {                                                                          \
        float *tmp = x0;                                                       \
        x0 = x;                                                                \
        x = tmp;                                                               \
    }

static float *d_u, *d_v, *d_u0, *d_v0, *d_x, *d_x0;

/* will be executed 1D with at least (N+2)^2 threads */
__global__ void cuda_add_source(int N, float *x, const float *s, float dt) {
    int size = (N + 2) * (N + 2);
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size) {
        x[i] += dt * s[i];
    }
}

static void call_cuda_add_source(int N, float *x, const float *s, float dt) {
    int size = (N + 2) * (N + 2);

    const dim3 blockSize(512, 1, 1);
    const dim3 gridSize((size + blockSize.x - 1) / blockSize.x, 1, 1);
    cuda_add_source<<<gridSize, blockSize>>>(N, x, s, dt);
}

static void add_source(int N, float *x, float *s, float dt) {
    int size = (N + 2) * (N + 2);
    for (int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

/* executed 2D with (N+2)x(N+2) */
/* this will be temporary (can just set bounds in other kernels) */
__global__ void cuda_set_bnd(int N, int b, float *x) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx == 0)
        x[IX(tx, ty)] = b == 1 ? -x[IX(1, ty)] : x[IX(1, ty)];
    if (tx == N + 1)
        x[IX(tx, ty)] = b == 1 ? -x[IX(N, ty)] : x[IX(N, ty)];
    if (ty == 0)
        x[IX(tx, ty)] = b == 2 ? -x[IX(tx, 1)] : x[IX(tx, 1)];
    if (ty == N + 1)
        x[IX(tx, ty)] = b == 2 ? -x[IX(tx, N)] : x[IX(tx, N)];

    __syncthreads();

    if (tx == 0 && ty == 0)
        x[IX(tx, ty)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
    if (tx == 0 && ty == N + 1)
        x[IX(tx, ty)] = 0.5f * (x[IX(1, N + 1)] + x[IX(0, N)]);
    if (tx == N + 1 && ty == 0)
        x[IX(tx, ty)] = 0.5f * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
    if (tx == N + 1 && ty == N + 1)
        x[IX(tx, ty)] = 0.5f * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

static void set_bnd(int N, int b, float *x) {
    for (int i = 1; i <= N; i++) {
        x[IX(0, i)]     = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
        x[IX(i, 0)]     = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
    }

    x[IX(0, 0)]         = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, N + 1)]     = 0.5f * (x[IX(1, N + 1)] + x[IX(0, N)]);
    x[IX(N + 1, 0)]     = 0.5f * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
    x[IX(N + 1, N + 1)] = 0.5f * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

/* executed 2D with N x N */
__global__ void cuda_lin_solve(int N, int b, float *x, const float *x0, float a, float c) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    // account for NxN --> (N+2)x(N+2)
    int i = tx + 1, j = ty + 1;

    if (tx < N && ty < N) {
        x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)])) / c;
    }
}

static void call_cuda_lin_solve(int N, int b, float *x, const float *x0, float a, float c) {
    for (int k = 0; k < 20; k++) {
        cuda_lin_solve<<<dim3((N+16-1)/16, (N+16-1)/16, 1), dim3(16, 16, 1)>>>(N, b, x, x0, a, c);
        cuda_set_bnd<<<dim3((N+2+16-1)/16, (N+2+16-1)/16, 1), dim3(16, 16, 1)>>>(N, b, x);
    }
}

static void lin_solve(int N, int b, float *x, float *x0, float a, float c) {
    for (int k = 0; k < 20; k++) {
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)])) / c;
            }
        }
        set_bnd(N, b, x);
    }
}

static void call_cuda_diffuse(int N, int b, float *x, float *x0, float diff, float dt) {
    float a = dt * diff * N * N;
    float c = 1 + 4 * a;

    call_cuda_lin_solve(N, b, x, x0, a, c);
}

static void diffuse(int N, int b, float *x, float *x0, float diff, float dt) {
    float a = dt * diff * N * N;
    lin_solve(N, b, x, x0, a, 1 + 4 * a);
}

/* execute 2D with NxN */
__global__ void cuda_advect(int N, int b, float *d, const float *d0, const float *u, const float *v, float dt) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    // account for NxN --> (N+2)x(N+2)
    int i = tx + 1, j = ty + 1;

    float dt0 = dt * N;
    if (tx < N && ty < N) {
        float x = i - dt0 * u[IX(i, j)];
        float y = j - dt0 * v[IX(i, j)];
        x = fminf(fmaxf(x, 0.5f), N + 0.5f);
        y = fminf(fmaxf(y, 0.5f), N + 0.5f);
        int i0 = floorf(x);
        int i1 = i0 + 1;
        int j0 = floorf(y);
        int j1 = j0 + 1;

        float s1 = x - i0;
        float s0 = 1 - s1;
        float t1 = y - j0;
        float t0 = 1 - t1;
        d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)])
                        + s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    }
}

static void call_cuda_advect(int N, int b, float *d, const float *d0, const float *u, const float *v, float dt) {
    cuda_advect<<<dim3((N+16-1)/16, (N+16-1)/16, 1), dim3(16, 16, 1)>>>(N, b, d, d0, u, v, dt);
    cuda_set_bnd<<<dim3((N+2+16-1)/16, (N+2+16-1)/16, 1), dim3(16, 16, 1)>>>(N, b, d);
}

static void advect(int N, int b, float *d, float *d0, float *u, float *v, float dt) {
    float dt0 = dt * N;

    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            float x = i - dt0 * u[IX(i, j)];
            float y = j - dt0 * v[IX(i, j)];
            if (x < 0.5f) {
                x = 0.5f;
            }
            if (x > N + 0.5f) {
                x = N + 0.5f;
            }
            int i0 = (int)x;
            int i1 = i0 + 1;
            if (y < 0.5f) {
                y = 0.5f;
            }
            if (y > N + 0.5f) {
                y = N + 0.5f;
            }
            int j0 = (int)y;
            int j1 = j0 + 1;
            float s1 = x - i0;
            float s0 = 1 - s1;
            float t1 = y - j0;
            float t0 = 1 - t1;
            d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)])
                          + s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        }
    }
    set_bnd(N, b, d);
}

__global__ void cuda_project0(int N, float *div, float *p, const float *u, const float *v) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    // account for NxN --> (N+2)x(N+2)
    int i = tx + 1, j = ty + 1;

    if (tx < N && ty < N) {
        div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]) / N;
        p[IX(i, j)] = 0;
    }
}

__global__ void cuda_project1(int N, float *u, float *v, const float *p) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    // account for NxN --> (N+2)x(N+2)
    int i = tx + 1, j = ty + 1;

    if (tx < N && ty < N) {
        u[IX(i, j)] -= 0.5f * N * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
        v[IX(i, j)] -= 0.5f * N * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
    }
}

static void call_cuda_project(int N, float *u, float *v, float *p, float *div) {
    cuda_project0<<<dim3((N+16-1)/16, (N+16-1)/16, 1), dim3(16, 16, 1)>>>(N, div, p, u, v);

    cuda_set_bnd<<<dim3((N+2+16-1)/16, (N+2+16-1)/16, 1), dim3(16, 16, 1)>>>(N, 0, div);
    cuda_set_bnd<<<dim3((N+2+16-1)/16, (N+2+16-1)/16, 1), dim3(16, 16, 1)>>>(N, 0, p);

    call_cuda_lin_solve(N, 0, p, div, 1, 4);

    cuda_project1<<<dim3((N+16-1)/16, (N+16-1)/16, 1), dim3(16, 16, 1)>>>(N, u, v, p);

    cuda_set_bnd<<<dim3((N+2+16-1)/16, (N+2+16-1)/16, 1), dim3(16, 16, 1)>>>(N, 1, u);
    cuda_set_bnd<<<dim3((N+2+16-1)/16, (N+2+16-1)/16, 1), dim3(16, 16, 1)>>>(N, 2, v);
}

static void project(int N, float *u, float *v, float *p, float *div) {
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]) / N;
            p[IX(i, j)] = 0;
        }
    }
    set_bnd(N, 0, div);
    set_bnd(N, 0, p);

    lin_solve(N, 0, p, div, 1, 4);

    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            u[IX(i, j)] -= 0.5f * N * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
            v[IX(i, j)] -= 0.5f * N * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
        }
    }
    set_bnd(N, 1, u);
    set_bnd(N, 2, v);
}

void cuda_dens_step(int N, float *x, float *x0, const float *u, const float *v, float diff, float dt) {
    int size = (N + 2) * (N + 2);
    int mem_size = size * sizeof(float);

    HANDLE_ERROR( cudaMemcpy(d_x, x, mem_size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(d_x0, x0, mem_size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(d_u, u, mem_size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(d_v, v, mem_size, cudaMemcpyHostToDevice) );

    // add_source(N, x, x0, dt);
    call_cuda_add_source(N, d_x, d_x0, dt);

    // SWAP(u0, u);
    SWAP(d_x0, d_x);

    // diffuse(N, 0, x, x0, diff, dt);
    call_cuda_diffuse(N, 0, d_x, d_x0, diff, dt);

    // SWAP(x0, x);
    SWAP(d_x0, d_x);

    // advect(N, 0, x, x0, u, v, dt);
    call_cuda_advect(N, 0, d_x, d_x0, d_u, d_v, dt);

    HANDLE_ERROR( cudaMemcpy(x, d_x, mem_size, cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(x0, d_x0, mem_size, cudaMemcpyDeviceToHost) );
}

void cuda_vel_step(int N, float *u, float *v, float *u0, float *v0, float visc, float dt) {
    int size = (N + 2) * (N + 2);
    int mem_size = size * sizeof(float);

    HANDLE_ERROR( cudaMemcpy(d_u, u, mem_size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(d_v, v, mem_size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(d_u0, u0, mem_size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(d_v0, v0, mem_size, cudaMemcpyHostToDevice) );

    // add_source(N, u, u0, dt);
    // add_source(N, v, v0, dt);
    call_cuda_add_source(N, d_u, d_u0, dt);
    call_cuda_add_source(N, d_v, d_v0, dt);

    // SWAP(u0, u);
    SWAP(d_u0, d_u);

    // diffuse(N, 1, u, u0, visc, dt);
    call_cuda_diffuse(N, 1, d_u, d_u0, visc, dt);

    // SWAP(v0, v);
    SWAP(d_v0, d_v);

    // diffuse(N, 2, v, v0, visc, dt);
    call_cuda_diffuse(N, 2, d_v, d_v0, visc, dt);

    // project(N, u, v, u0, v0);
    call_cuda_project(N, d_u, d_v, d_u0, d_v0);

    // SWAP(u0, u);
    // SWAP(v0, v);
    SWAP(d_u0, d_u);
    SWAP(d_v0, d_v);

    // advect(N, 1, u, u0, u0, v0, dt);
    // advect(N, 2, v, v0, u0, v0, dt);
    call_cuda_advect(N, 1, d_u, d_u0, d_u0, d_v0, dt);
    call_cuda_advect(N, 2, d_v, d_v0, d_u0, d_v0, dt);

    // project(N, u, v, u0, v0);
    call_cuda_project(N, d_u, d_v, d_u0, d_v0);

    HANDLE_ERROR( cudaMemcpy(u, d_u, mem_size, cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(u0, d_u0, mem_size, cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(v, d_v, mem_size, cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(v0, d_v0, mem_size, cudaMemcpyDeviceToHost) );
}

void cuda_init(int N) {
    int size = (N + 2) * (N + 2) * sizeof(float);

    cudaMalloc((void **) &d_u, size);
    cudaMalloc((void **) &d_v, size);
    cudaMalloc((void **) &d_u0, size);
    cudaMalloc((void **) &d_v0, size);
    cudaMalloc((void **) &d_x, size);
    cudaMalloc((void **) &d_x0, size);

    cudaMemset(&d_u, 0, size);
    cudaMemset(&d_v, 0, size);
    cudaMemset(&d_u0, 0, size);
    cudaMemset(&d_v0, 0, size);
    cudaMemset(&d_x, 0, size);
    cudaMemset(&d_x0, 0, size);
}

void cuda_cleanup() {
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_u0);
    cudaFree(d_v0);
    cudaFree(d_x);
    cudaFree(d_x0);
}

/*
 * CUDA Fluid Solver
 * Sam Freed & Kyle Mulligan
 *
 * 2D fluid solver written for Cal Poly CSC 515, Fall 2017.
 * Based on Jos Stam's GDC2003 demo from "Real-Time Fluid Dynamics for Games"
 */

#include <GL/glut.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

/* macros */
#define IX(i, j) ((i) + (N + 2) * (j))

#define FRAME_STOP 4
#define DEFAULT_N 1024
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800

#define USE_CUDA
#define USE_PINNED

/* external definitions (from solver.c) */
extern void dens_step(int N, float *x, float *x0, float *u, float *v,
                      float diff, float dt);
extern void vel_step(int N, float *u, float *v, float *u0, float *v0,
                     float visc, float dt);

/* extern void cuda_dens_step(int N, float *x, float *x0, const float *u, const float *v, */
/*                       float diff, float dt); */
/* extern void cuda_vel_step(int N, float *u, float *v, float *u0, float *v0, */
/*                      float visc, float dt); */
extern void cuda_init(int N);
extern void cuda_update(float *milliseconds, int N, float *x, float *x0, float *u, float *v, float *u0, float *v0, float diff, float visc, float dt);
extern void cuda_cleanup();

static void pre_display();
static void post_display();
static void key_func(unsigned char key, int x, int y);
static void mouse_func(int button, int state, int x, int y);
static void motion_func(int x, int y);
static void reshape_func(int width, int height);
static void idle_func();
static void display_func();
static void open_glut_window();

/* global variables */
static int dvel;

static int win_id;
static int win_x, win_y;
static int mouse_down[3];
static int omx, omy, mx, my;

static int frame = 0;

class FluidSolver {
public:
    FluidSolver()
        : FluidSolver(64, 0.1f, 0.0f, 0.0f, 5.0f, 100.0f)
    {}

    FluidSolver(int N, float dt, float diff, float visc, float force, float source)
        : N(N), dt(dt), diff(diff), visc(visc), force(force), source(source)
    {
        int size = (N + 2) * (N + 2);

#ifdef USE_PINNED
        cudaMallocHost((void **)&u, size * sizeof(float));
        cudaMallocHost((void **)&v, size * sizeof(float));
        cudaMallocHost((void **)&u_prev, size * sizeof(float));
        cudaMallocHost((void **)&v_prev, size * sizeof(float));
        cudaMallocHost((void **)&dens, size * sizeof(float));
        cudaMallocHost((void **)&dens_prev, size * sizeof(float));
#else
        u = new float[size];
        v = new float[size];
        u_prev = new float[size];
        v_prev = new float[size];
        dens = new float[size];
        dens_prev = new float[size];
#endif

        clear();
    }

    virtual ~FluidSolver() {
#ifdef USE_PINNED
        cudaFreeHost(u);
        cudaFreeHost(v);
        cudaFreeHost(u_prev);
        cudaFreeHost(v_prev);
        cudaFreeHost(dens);
        cudaFreeHost(dens_prev);
#else
        delete u;
        delete v;
        delete u_prev;
        delete v_prev;
        delete dens;
        delete dens_prev;
#endif
        printf("\n\nAverage frame time: %5.2f ms\n", frametime / frame);
    }

    void clear() {
        int size = (N + 2) * (N + 2);

        for (int i = 0; i < size; i++) {
            u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
        }
    }

    void addSource(int i, int j) {
        if (i < 1 || i > N || j < 1 || j > N) {
            return;
        }

        dens_prev[IX(i, j)] = source;
    }

    void addForce(int i, int j, float force_u, float force_v) {
        if (i < 1 || i > N || j < 1 || j > N) {
            return;
        }

        u[IX(i, j)] = force * force_u;
        v[IX(i, j)] = force * force_v;
    }

    const float *const getU() const { return u; }
    const float *const getV() const { return v; }
    const float *const getDensity() const { return dens; }

    void draw_velocity() const {
        float h = 1.0f / N;

        glColor3f(1.0f, 1.0f, 1.0f);
        glLineWidth(1.0f);

        glBegin(GL_LINES);

        for (int i = 1; i <= N; i++) {
            float x = (i - 0.5f) * h;
            for (int j = 1; j <= N; j++) {
                float y = (j - 0.5f) * h;

                glVertex2f(x, y);
                glVertex2f(x + u[IX(i, j)], y + v[IX(i, j)]);
            }
        }

        glEnd();
    }
    void draw_density() const {
        float h = 1.0f / N;

        glBegin(GL_QUADS);

        for (int i = 0; i <= N; i++) {
            float x = (i - 0.5f) * h;
            for (int j = 0; j <= N; j++) {
                float y = (j - 0.5f) * h;

                float d00 = dens[IX(i, j)];
                float d01 = dens[IX(i, j + 1)];
                float d10 = dens[IX(i + 1, j)];
                float d11 = dens[IX(i + 1, j + 1)];

                glColor3f(d00, d00, d00);
                glVertex2f(x, y);
                glColor3f(d10, d10, d10);
                glVertex2f(x + h, y);
                glColor3f(d11, d11, d11);
                glVertex2f(x + h, y + h);
                glColor3f(d01, d01, d01);
                glVertex2f(x, y + h);
            }
        }

        glEnd();
    }

    void get_from_UI() {
        float *d = dens_prev, *u = u_prev, *v = v_prev;
        int i, j, size = (N + 2) * (N + 2);

        for (i = 0; i < size; i++) {
            u[i] = v[i] = d[i] = 0.0f;
        }

        if (!mouse_down[0] && !mouse_down[2])
            return;

        i = (int)((mx / (float)win_x) * N + 1);
        j = (int)(((win_y - my) / (float)win_y) * N + 1);

        if (i < 1 || i > N || j < 1 || j > N)
            return;

        if (mouse_down[0]) {
            u[IX(i, j)] = force * (mx - omx);
            v[IX(i, j)] = force * (omy - my);
        }

        if (mouse_down[2]) {
            d[IX(i, j)] = source;
        }

        omx = mx;
        omy = my;
    }

    virtual void update() = 0;

protected:
    int N = 64;
    float dt = 0.1f, diff = 0.0f, visc = 0.0f;
    float force = 5.0f, source = 100.0f;

    float *u, *v, *u_prev, *v_prev;
    float *dens, *dens_prev;

    float frametime = 0.0f;
};

class SerialFluidSolver : public FluidSolver {
public:
    SerialFluidSolver()
        : SerialFluidSolver(64, 0.1f, 0.0f, 0.0f, 5.0f, 100.0f)
    {}

    SerialFluidSolver(int N, float dt, float diff, float visc, float force, float source)
        : FluidSolver(N, dt, diff, visc, force, source)
    {}

    void update() override {
        auto start = std::chrono::high_resolution_clock::now();
        vel_step(N, u, v, u_prev, v_prev, visc, dt);
        dens_step(N, dens, dens_prev, u, v, diff, dt);
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = stop - start;
        printf("\rUpdate took %5.2f ms", elapsed.count() * 1000);

        frametime += elapsed.count() * 1000;
    }
};

class CudaFluidSolver : public FluidSolver {
public:
    CudaFluidSolver()
        : CudaFluidSolver(64, 0.1f, 0.0f, 0.0f, 5.0f, 100.0f)
    {}

    CudaFluidSolver(int N, float dt, float diff, float visc, float force, float source)
        : FluidSolver(N, dt, diff, visc, force, source)
    {
        cuda_init(N);
    }

    ~CudaFluidSolver() {
        cuda_cleanup();
    }

    void update() override {
        float milliseconds;
        cuda_update(&milliseconds, N, dens, dens_prev, u, v, u_prev, v_prev, diff, visc, dt);

        printf("\rUpdate took %5.2f ms", milliseconds);
        frametime += milliseconds;
    }
};

/*
  ----------------------------------------------------------------------
   main --- main routine
  ----------------------------------------------------------------------
*/
FluidSolver *solver = nullptr;
int main(int argc, char **argv) {
    glutInit(&argc, argv);

    int N;
    float dt, diff, visc, force, source;
    if (argc != 1 && argc != 7) {
        fprintf(stderr, "usage : %s N dt diff visc force source\n",
            "where:\n"
            "\t N      : grid resolution\n"
            "\t dt     : time step\n"
            "\t diff   : diffusion rate of the density\n"
            "\t visc   : viscosity of the fluid\n"
            "\t force  : scales the mouse movement that generate a force\n"
            "\t source : amount of density that will be deposited\n", argv[0]
        );
        exit(1);
    }

    if (argc == 1) {
        N = DEFAULT_N;
        dt = 0.1f;
        diff = 0.0f;
        visc = 0.0f;
        force = 5.0f;
        source = 100.0f;
        fprintf(stderr,
                "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g "
                "source=%g\n",
                N, dt, diff, visc, force, source);
    }
    else {
        N = atoi(argv[1]);
        dt = atof(argv[2]);
        diff = atof(argv[3]);
        visc = atof(argv[4]);
        force = atof(argv[5]);
        source = atof(argv[6]);
    }

    // Print device information (from https://devblogs.nvidia.com/parallelforall/how-query-device-properties-and-handle-errors-cuda-cc/)
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

    printf("\n\nHow to use this demo:\n\n"
        "\t Add densities with the right mouse button\n"
        "\t Add velocities with the left mouse button and dragging the mouse\n"
        "\t Toggle density/velocity display with the 'v' key\n"
        "\t Clear the simulation by pressing the 'c' key\n"
        "\t Quit by pressing the 'q' key\n"
    );

    dvel = 0;

#ifdef USE_CUDA
    solver = new CudaFluidSolver(N, dt, diff, visc, force, source);
#else
    solver = new SerialFluidSolver(N, dt, diff, visc, force, source);
#endif

    win_x = WINDOW_WIDTH;
    win_y = WINDOW_HEIGHT;
    open_glut_window();

    glutMainLoop();

    exit(0);
}

/*
  ----------------------------------------------------------------------
   OpenGL specific drawing routines
  ----------------------------------------------------------------------
*/

static void pre_display(void) {
    glViewport(0, 0, win_x, win_y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

static void post_display(void) { glutSwapBuffers(); }

/*
  ----------------------------------------------------------------------
   GLUT callback routines
  ----------------------------------------------------------------------
*/

static void key_func(unsigned char key, int x, int y) {
    switch (key) {
    case 'c':
    case 'C':
        solver->clear();
        break;

    case 'q':
    case 'Q':
        delete solver;
        exit(0);
        break;

    case 'v':
    case 'V':
        dvel = !dvel;
        break;
    }
}

static void mouse_func(int button, int state, int x, int y) {
    omx = mx = x;
    omx = my = y;

    mouse_down[button] = state == GLUT_DOWN;
}

static void motion_func(int x, int y) {
    mx = x;
    my = y;
}

static void reshape_func(int width, int height) {
    glutSetWindow(win_id);
    glutReshapeWindow(width, height);

    win_x = width;
    win_y = height;
}

static void idle_func(void) {
    solver->get_from_UI();
    solver->update();

    glutSetWindow(win_id);
    glutPostRedisplay();

    frame++;
    if (FRAME_STOP && frame == FRAME_STOP) {
        delete solver;
        exit(0);
    }
}

static void display_func(void) {
    pre_display();

    if (dvel)
        solver->draw_velocity();
    else
        solver->draw_density();

    post_display();
}

/*
  ----------------------------------------------------------------------
   open_glut_window --- open a glut compatible window and set callbacks
  ----------------------------------------------------------------------
*/

static void open_glut_window(void) {
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

    glutInitWindowPosition(0, 0);
    glutInitWindowSize(win_x, win_y);
    win_id = glutCreateWindow("CUDA FluidSim");

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();

    pre_display();

    glutKeyboardFunc(key_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutReshapeFunc(reshape_func);
    glutIdleFunc(idle_func);
    glutDisplayFunc(display_func);
}

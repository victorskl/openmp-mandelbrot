// COMP90025 Project 1A: OpenMP and Mandelbrot Set
// San Kho Lin (829463) sanl1@student.unimelb.edu.au

// Compile with:
//    -DRANDOM  - Static random task assignment
//    -DSTATIC  - OpenMP schedule(static)
//    -DGUIDED  - OpenMP schedule(guided)
//    none      - OpenMP schedule(dynamic) - This is default

// Tested with GCC:
//  gcc -fopenmp mandelbrot.cpp -o mandelbrot.exe
//  gcc -fopenmp mandelbrot.cpp -o mandelbrot.exe -DRANDOM

// Tested with Intel Compiler on Windows:
//  icl /Qopenmp /Qstd=c99 /Tc mandelbrot.cpp -o mandelbrot.exe
//  icl /Qopenmp /Qstd=c99 /Tc mandelbrot.cpp -o mandelbrot.exe -DRANDOM

// Run with:
//  ./mandelbrot.exe -2.0 1.0 -1.0 1.0 100 10000 -1 1.0 0.0 1.0 100 10000

// Intel Compiler icl is sensitive on header include
// and file extension. Without /Tc option, the extension
// of source file (.c .cpp) affects on how it is compiled.
// GCC has no such issue.

// Assignment require submit in filename 'mandelbrot.cpp'
// quote "a single file with exactly that name".
// Propose filename should be 'mandelbrot.c'

// Since the source code is mainly C, the following header
// rename to C sytle header.
//#include <cstdio>
//#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <complex.h>

#ifdef RANDOM
#include <time.h>
typedef struct {
  double real, img;
} Coord;

// Reference:
// Dr. Norm Matloff [Online eBook] "Programming on Parallel Machines"
// Chapter 2.2 Load Balancing
// Chapter 2.4 Static (But Possibly Random) Task Assignment Better Than Dynamic
// Chapter 4.4 Example: Mandelbrot Set
void findmyrange (int n, int nth, int me, int *myrange) {
    int chunksize = n / nth;
    myrange[0] = me * chunksize;
    if (me < nth-1) {
        myrange[1] = (me+1) * chunksize - 1;
    } else {
        myrange[1] = n - 1;
    }
}
#endif

// return 1 if in set, 0 otherwise
int inset(double complex c, int maxiter) {
    double z_real, z_img;
    double complex
    z = c;
    for (int iters = 0; iters < maxiter; iters++) {
        z = z * z + c;
        z_real = creal(z);
        z_img = cimag(z);
        if (z_real * z_real + z_img * z_img > 4.0) return 0;
    }
    return 1;
}

// count the number of points in the set, within the region
int mandelbrotSetCount(double real_lower, double real_upper, double img_lower, double img_upper, int num, int maxiter) {
    int count = 0;

    #ifdef RANDOM

    int idx_size = (num+1) * (num+1);
    Coord points[idx_size];
    double real_step = (real_upper-real_lower)/num;
    double img_step = (img_upper-img_lower)/num;
    int cnt = 0;
    for (int real=0; real<=num; real++) {
        for (int img=0; img<=num; img++) {
            Coord p;
            p.real = real_lower+real*real_step;
            p.img = img_lower+img*img_step;
            points[cnt] = p;
            cnt++;
        }
    }

    // Reference:
    // Fisher–Yates shuffle
    int n = sizeof(points) / sizeof(points[0]);
    srand(time(NULL));
    for (int i = n-1; i > 0; i--) {
        int j = rand() % (i+1);
        Coord temp = points[i];
        points[i] = points[j];
        points[j] = temp;
    }

    #pragma omp parallel reduction(+:count)
    {
        double complex c;
        int myrange[2];
        int me = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int i;
        findmyrange(idx_size, nth, me, myrange);
        for (i = myrange[0]; i <= myrange[1]; i++) {
            Coord p = points[i];
            c = p.real + p.img*I;
            count += inset(c, maxiter);
        }
    }

    #else

    #pragma omp parallel
    {
        double complex c;
        double real_step = (real_upper - real_lower) / num;
        double img_step = (img_upper - img_lower) / num;

        #ifdef STATIC
        #pragma omp for reduction(+:count) schedule(static)
        #elif defined GUIDED
        #pragma omp for reduction(+:count) schedule(guided)
        #else
        #pragma omp for reduction(+:count) schedule(dynamic)
        #endif

        for (int real = 0; real <= num; real++) {
            for (int img = 0; img <= num; img++) {
                c = (real_lower + real * real_step) + (img_lower + img * img_step) * I;
                count += inset(c, maxiter);
            }
        }
    }

    #endif

    return count;
}

// main
int main(int argc, char *argv[]) {
    double real_lower;
    double real_upper;
    double img_lower;
    double img_upper;
    int num;
    int maxiter;
    int num_regions = (argc - 1) / 6;
    for (int region = 0; region < num_regions; region++) {
        // scan the arguments
        sscanf(argv[region * 6 + 1], "%lf", &real_lower);
        sscanf(argv[region * 6 + 2], "%lf", &real_upper);
        sscanf(argv[region * 6 + 3], "%lf", &img_lower);
        sscanf(argv[region * 6 + 4], "%lf", &img_upper);
        sscanf(argv[region * 6 + 5], "%i", &num);
        sscanf(argv[region * 6 + 6], "%i", &maxiter);
        printf("%d\n", mandelbrotSetCount(real_lower, real_upper, img_lower, img_upper, num, maxiter));
    }
    return EXIT_SUCCESS;
}

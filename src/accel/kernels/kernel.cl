#ifdef ALTERA_CL
__attribute__((reqd_work_group_size(1,1,1)))
__attribute__((num_simd_work_items(1)))
__attribute__((num_compute_units(1)))
#endif
__kernel void vector_add(
        __global float const * A,
        __global float const * B,
        __global float *C
) {
  // Get the index of the current element
  int i = get_global_id(0);

  // Do the operation
  C[i] = A[i] + B[i];
}

#ifdef ALTERA_CL
__attribute__((reqd_work_group_size(1,1,1)))
__attribute__((num_simd_work_items(1)))
__attribute__((num_compute_units(1)))
#endif
__kernel void GS1(
        __global float *m_dev,
        __global float *a_dev,
        __global float *b_dev,
        const int size,
        const int t
) {
    const int gx = get_global_id(0);

    if (gx < size-1-t) {
        *(m_dev + size * (gx + t + 1) + t) = *(a_dev + size * (gx + t + 1) + t) / *(a_dev + size * t + t);
    }
}


#ifdef ALTERA_CL
__attribute__((reqd_work_group_size(1,1,1)))
__attribute__((num_simd_work_items(1)))
__attribute__((num_compute_units(1)))
#endif
__kernel void GS2(
        __global float *m_dev,
        __global float *a_dev,
        __global float *b_dev,
        const int size,
        const int t
) {
    const int gidx = get_global_id(0);
    const int gidy = get_global_id(1);
    if (gidx < size-1-t && gidy < size-t) {
        a_dev[size*(gidx+1+t) + (gidy+t)] -= m_dev[size*(gidx + 1+t)+t] * a_dev[size*t+(gidy+t)];

        if(gidy == 0) {
            b_dev[gidx+1+t] -= m_dev[size*(gidx+1+t)+(gidy+t)] * b_dev[t];
        }
    }
}

#ifdef ALTERA_CL
__attribute__((reqd_work_group_size(1,1,1)))
__attribute__((num_simd_work_items(1)))
__attribute__((num_compute_units(1)))
#endif
__kernel void mult_2x2block_diag(
        const int N,
        __constant float const * D,
        __global float const * x,
        __global float * y
) {
    // Get the current block
    const int gidx = get_global_id(0);

    // Perform a 2x2 block multiplication
    if (gidx < N) {
#define D(i, j) D[i * 2 + j]
        const int i = gidx;
        float f = 0.;
        int r = (i / 2) * 2;
        for (int j = r; j < r + 2; j++) {
            f += D(i, j) * x[j];
        }
        y[i] = f;
#undef D
    }
}

/*
Hard element-wise thesholding on a row vector, where the thresholds are also a
vector

params:
    N: int       // Number of elements
    L: &float[N] // Saturation lower bound
    U: &float[N] // Saturation upper bound
    A: &float[N] // Vector to be thresholded
*/
#ifdef ALTERA_CL
__attribute__((reqd_work_group_size(1,1,1)))
__attribute__((num_simd_work_items(1)))
__attribute__((num_compute_units(1)))
#endif
__kernel void hard_threshold_vector(
        const int N,
        __constant float const * restrict L,
        __constant float const * restrict U,
        __global float * restrict A
) {
    const int gidx = get_global_id(0);
    if (gidx < N) {
        const float l = L[gidx];
        const float u = U[gidx];
        const float x = A[gidx];

        // threshold
        A[gidx] = (x > u ? u :
                  (x < l ? l : x));
    }
}

/*
Hard element-wise thesholding on a row vector

params:
    N: int       // Number of elements
    L: float     // Saturation lower bound
    U: float     // Saturation upper bound
    A: &float[N] // Vector to be thresholded
*/
#ifdef ALTERA_CL
__attribute__((reqd_work_group_size(1,1,1)))
__attribute__((num_simd_work_items(1)))
__attribute__((num_compute_units(1)))
#endif
__kernel void hard_threshold(
        const int N,
        const float L,
        const float U,
        __global float * restrict A
) {
    const int gidx = get_global_id(0);
    if (gidx < N) {
        const float x = A[gidx];

        // threshold
        A[gidx] = (x > U ? U :
                  (x < L ? L : x));
    }
}

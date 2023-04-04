#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    return mat->data[row*(mat->cols) + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    mat->data[row*(mat->cols) + col] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return -1;

    *mat = malloc(sizeof(matrix));
    if (*mat == NULL) return -2;

    (*mat)->data = calloc(rows*cols, sizeof(double));
    if ((*mat)->data == NULL) return -2;

    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->parent = NULL;
    (*mat)->ref_cnt = 1;

    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    if (mat == NULL) return;

    if (mat->parent == NULL) {
        if ((mat->ref_cnt)-- == 1) {
            free(mat->data);
            free(mat);
        }
    } else {
        deallocate_matrix(mat->parent);
        free(mat);
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return -1;

    *mat = malloc(sizeof(matrix));
    if (*mat == NULL) return -2;
    (*mat)->data = from->data + offset;
    (*mat)->parent = from;
    (*mat)->rows = rows;
    (*mat)->cols = cols;

    (from->ref_cnt)++;
    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            mat->data[i*mat->cols + j] = val;
        }
    }
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            double val = mat->data[i*mat->cols + j];
            if (val < 0) result->data[i*mat->cols + j] = -val;
            else result->data[i*mat->cols + j] = val;
        }
    }

    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            double val = mat->data[i*mat->cols + j];
            result->data[i*mat->cols + j] = -val;
        }
    }

    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            int idx = i*result->cols + j;
            result->data[idx] = mat1->data[idx] + mat2->data[idx];
        }
    }

    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            int idx = i*result->cols + j;
            result->data[idx] = mat1->data[idx] - mat2->data[idx];
        }
    }

    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    matrix* mat2_T = NULL;
    allocate_matrix(&mat2_T, mat2->cols, mat2->rows);
    transpose_matrix(mat2_T, mat2);

    mul_matrix_helper(result, mat1, mat2_T);
    deallocate_matrix(mat2_T);
    return 0;
}

void mul_matrix_helper(matrix *result, matrix *mat1, matrix *mat2) {
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < mat2->rows; j++) {
            double prod = 0.0;
            for (int k = 0; k < mat1->cols; k++) {
                int m1 = i * mat1->cols + k;
                int m2 = j * mat2->cols + k;
                prod += mat1->data[m1] * mat2->data[m2];
            }
            int r = i * result->cols + j;
            result->data[r] = prod;
        }
    }
}

void transpose_matrix(matrix *result, matrix* mat) {
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            int r = i*result->cols + j;
            int m = j*mat->cols + i;
            result->data[r] = mat->data[m];
        }   
    }
}

void identity_matrix(matrix *result) {
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            int idx = i*result->cols + j;
            if (i == j) result->data[idx] = 1.0;
            else result->data[idx] = 0.0;
        }   
    }
}

void copy_matrix(matrix* result, matrix* mat) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            int idx = i*mat->cols + j;
            result->data[idx] = mat->data[idx];
        }
    }
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    if (pow == 0) {
        identity_matrix(result);
    } else if (pow == 1) {
        copy_matrix(result, mat);
    } else {
        matrix* mat_T = NULL;
        allocate_matrix(&mat_T, mat->cols, mat->rows);
        transpose_matrix(mat_T, mat);

        mul_matrix_helper(result, mat, mat_T);

        if (pow > 2) {
            matrix* aux = NULL;
            allocate_matrix(&aux, result->rows, result->cols);

            for (int i = 2; i < pow; i++) {
                if (i%2 == 0) mul_matrix_helper(aux, result, mat_T);
                else mul_matrix_helper(result, aux, mat_T);
            }

            if (pow%2 != 0) copy_matrix(result, aux);
            deallocate_matrix(aux);
        }

        deallocate_matrix(mat_T);
    }
    return 0;
}

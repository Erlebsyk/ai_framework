/**
 * \file	matrix_test.cuh
 * \brief	
 * 
 * \author	Erlend Isachsen (erlend.isachsen@7sense.no)
 * 
*/

#ifndef MATRIX_CUH_
#define MATRIX_CUH_

/** Standard library header includes	*/
#include <cstdint>
#include <ostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <array>

/** NRF / Zephyr header includes	*/


/** Project header includes	*/

/**
 * \brief Get the matrix dimensions resulting from the dot product A*B.
 */


namespace utils
{
	/** Variable declarations	*/
	typedef const std::initializer_list<std::initializer_list<float>> matrix_init_list_t;
	template <size_t N, size_t M> class DMatrix;

	/** Funciton declarations	*/
	#define MATRIX_DOTP_DIM(A, B) B.kColsN, A.kRowsN

	void cuda_error_parser(
		std::string caller_f,
		std::string called_f,
		cudaError_t error,
		std::string context = ""
	);
	
	void cublas_error_parser(
		std::string caller_f,
		std::string called_f,
		cublasStatus_t status,
		std::string context = ""
	);

	/** Classes, structs and enums	*/

	template <size_t N, size_t M>
	class DMatrix
	{
	protected:
		float* mat_;
		cublasHandle_t cublas_handle_;

	public:
		static const size_t kColsN = N;
		static const size_t kRowsN = M;
		static const size_t kLength = N * M;
		static const size_t kSize = N * M * sizeof(float);

		__host__ __device__ DMatrix();
		__host__ __device__ DMatrix(const DMatrix &other);
		__host__ __device__ DMatrix(DMatrix &&other) noexcept;
		__host__ __device__ DMatrix(matrix_init_list_t &matrix);
		__host__ __device__ ~DMatrix() noexcept;

		__host__ __device__ float At(size_t col_i, size_t row_i) const;
		__host__ __device__ void Set(size_t col_i, size_t row_i, float val);

		__host__ __device__ const float* Get() const;

		__host__ __device__ float SumGet() const;
		__host__ __device__ float SumRowGet(size_t row_i) const;
		__host__ __device__ float SumColGet(size_t col_i) const;

		void NormalizeFull();
		void NormalizeByRow();
		void NormalizeByColumn();

		__host__ __device__ bool operator==(const DMatrix<N, M> &other) const;

		__host__ __device__ DMatrix &operator=(const DMatrix& other);
		__host__ __device__ DMatrix &operator=(matrix_init_list_t&matrix);
		__host__ __device__ DMatrix &operator-();
		
		__host__ __device__ DMatrix &operator+=(const DMatrix &rhs);
		
		__host__ __device__ DMatrix& operator*= (float d_rhs);
		__host__ __device__ DMatrix &operator*= (const float* d_rhs);
	};	

	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N,M> operator+(DMatrix<N, M> lhs, const DMatrix<N, M>& rhs);
	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M> operator+(DMatrix<N, M> lhs, const DMatrix<N, 1>& rhs);
	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M> operator+(DMatrix<N, M> lhs, const DMatrix<1, M>& rhs);

	template <size_t N, size_t M, size_t P>
	__host__ __device__ DMatrix<P, M> operator*(const DMatrix<N,M> &lhs, const DMatrix<P,N> &rhs);

	template <size_t N, size_t M>
	std::ostream& operator<<(std::ostream& os, const DMatrix<N,M>& m);

	/** Funciton Implementations	 */


} // Namespace ai

#include "matrix.cu"

#endif // MATRIX_CUH_

/*
 *	--- End of file ---
 */
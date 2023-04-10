/**
 * \file	matrix_test.cu
 * \brief	
 * 
 * \author	Erlend Isachsen (erlend.isachsen@7sense.no)
 * 
*/

#ifndef MATRIX_CU_
#define MATRIX_CU_

#include "matrix.cuh"

#ifndef MATRIX_CUH_
#error __FILE__ should only be included from 'matrix.cuh'.
#endif // MATRIX_CUH_

/** Standard library header includes	*/
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <sstream>
#include <assert.h>

/** Project header includes	*/

namespace utils
{
	/** Variable declarations	*/

	/** Funciton declarations	*/
	

	/** Classes, structs and enums	*/


	/** Funciton Implementations	 */
	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M>::DMatrix() :
		mat_ { nullptr },
		cublas_handle_{}
	{
		// Allocate the matrix buffer on the device.
#ifndef __CUDA__ARCH__
		cudaError_t cuda_err = cudaMalloc(&mat_, kSize);
		cuda_error_parser(__FUNCSIG__, "cudaMalloc", cuda_err, "mat_");
#else
		mat_ = (float*)malloc(kSize);
#endif
		
		// Create the cublas handle structure
		const cublasStatus_t cublas_err = cublasCreate(&cublas_handle_);
		assert(CUBLAS_STATUS_SUCCESS == cublas_err);
	}

	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M>::DMatrix(const DMatrix<N, M> &other) :
		DMatrix<N,M>()
	{
#ifndef __CUDA__ARCH__
		const cudaError_t cuda_err = cudaMemcpy(mat_, other.Get(), kSize, cudaMemcpyDeviceToDevice);
		cuda_error_parser(__FUNCSIG__, "cudaMemcpy", cuda_err, "other.mat_ -> mat_");
#else
		memcpy(mat_, other.Get(), kSize);
#endif
	}

	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M>::DMatrix(DMatrix<N, M> &&other) noexcept :
		mat_{ other.mat_ },
		cublas_handle_	{ other.cublas_handle_	}
	{
		other.mat_ = nullptr;
		other.cublas_handle_ = nullptr;
	}

	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M>::DMatrix(matrix_init_list_t &matrix) :
		DMatrix<N, M>()
	{
		assert(matrix.size() == kRowsN);

		size_t row_i = 0;
		size_t col_i = 0;
		for (const std::initializer_list<float>& row : matrix)
		{
			assert(row.size() == kColsN);
			col_i = 0;
			for (const float &value : row)
			{
				Set(col_i, row_i, value);
				col_i++;
			}
			row_i++;
		}
	}

	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M>::~DMatrix() noexcept
	{
		if(mat_ != nullptr)
		{
#ifndef __CUDA__ARCH__
			cudaFree(mat_);
#else
			free(mat_);
#endif
			mat_ = nullptr;
		}
		if(cublas_handle_ != nullptr)
		{
			cublasDestroy(cublas_handle_);
			cublas_handle_ = nullptr;
		}
	}

	template <size_t N, size_t M>
	__host__ __device__ float DMatrix<N, M>::At(const size_t col_i, const size_t row_i) const
	{
#ifndef __CUDA__ARCH__
		float h_val;
		const cudaError_t cuda_err = cudaMemcpy(&h_val, &(mat_[kRowsN * col_i + row_i]), sizeof(float), cudaMemcpyDeviceToHost);
		cuda_error_parser(__FUNCSIG__, "cudaMemcpy", cuda_err, "mat_[kRowsN * col_i + row_i] -> h_val");
		return h_val;
#else
		return mat_[kRowsN * col_i + row_i];
#endif
	}

	template <size_t N, size_t M>
	__host__ __device__ void DMatrix<N, M>::Set(const size_t col_i, const size_t row_i, const float val)
	{
#ifndef __CUDA__ARCH__
		const cudaError_t cuda_err = cudaMemcpy(&(mat_[kRowsN * col_i + row_i]), &val, sizeof(float), cudaMemcpyHostToDevice);
		cuda_error_parser(__FUNCSIG__, "cudaMemcpy", cuda_err, "val -> mat_[kRowsN * col_i + row_i]");
#else
		mat_[kRowsN * col_i + row_i] = val;
#endif
	}

	template <size_t N, size_t M>
	__host__ __device__ const float* DMatrix<N, M>::Get() const
	{
		return mat_;
	}

	template <size_t N, size_t M>
	__host__ __device__ float DMatrix<N, M>::SumGet() const
	{
		float sum;
		const cublasStatus_t cublas_err = cublasSasum(cublas_handle_, kLength, mat_, 1, &sum);
		assert(cublas_err == CUBLAS_STATUS_SUCCESS);
		return sum;
	}

	template <size_t N, size_t M>
	__host__ __device__ float DMatrix<N, M>::SumRowGet(const size_t row_i) const
	{
		// Boundary check
		assert(row_i < kRowsN);

		// Obtain the sum
		float row_sum;
		const cublasStatus_t cublas_err = cublasSasum(cublas_handle_, kColsN, mat_ + row_i, kRowsN, &row_sum);
		assert(cublas_err == CUBLAS_STATUS_SUCCESS);

		return row_sum;
	}

	template <size_t N, size_t M>
	__host__ __device__ float DMatrix<N, M>::SumColGet(const size_t col_i) const
	{
		// Boundary checl
		assert(col_i < kColsN);

		// Obtain the sum
		float col_sum;
		const cublasStatus_t cublas_err = cublasSasum(cublas_handle_, kRowsN, mat_ + col_i * kRowsN, 1, &col_sum);
		assert(cublas_err == CUBLAS_STATUS_SUCCESS);

		return col_sum;
	}

	template <size_t N, size_t M>
	void DMatrix<N, M>::NormalizeFull()
	{
		*this *= (1.0f / SumGet());
	}

	template <size_t N, size_t M>
	void DMatrix<N, M>::NormalizeByRow()
	{
		float row_factor;
		cublasStatus_t cublas_err;
		for (size_t i = 0; i < kRowsN; i++)
		{
			row_factor = 1.0f / SumRowGet(i);
			cublas_err = cublasSscal(cublas_handle_, kColsN, &row_factor, mat_ + i, kRowsN);
			cublas_error_parser(__FUNCSIG__, "cublasSscal", cublas_err);
		}
	}

	template <size_t N, size_t M>
	void DMatrix<N, M>::NormalizeByColumn()
	{
		float col_factor;
		cublasStatus_t cublas_err;
		for (size_t i = 0; i < kColsN; i++)
		{
			col_factor = 1.0f / SumColGet(i);
			cublas_err = cublasSscal(cublas_handle_, kRowsN, &col_factor, mat_ + i * kRowsN, 1);
			cublas_error_parser(__FUNCSIG__, "cublasSscal", cublas_err);
		}
	}

	template <size_t N, size_t M>
	__host__ __device__ bool DMatrix<N, M>::operator==(const DMatrix<N, M>& other) const
	{
		bool equal = true;
		for (size_t col_i = 0; col_i < N && (equal); col_i++)
		{
			for (size_t row_i = 0; (row_i < M) && (equal); row_i++)
			{
				if (At(col_i, row_i) - other.At(col_i, row_i) > FLT_EPSILON)
					equal = false;
			}
		}

		return equal;
	}

	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M> &DMatrix<N, M>::operator=(const DMatrix<N, M> &other)
	{
#ifndef __CUDA__ARCH__
		const cudaError_t cuda_err = cudaMemcpy(mat_, other.Get(), kSize, cudaMemcpyDeviceToDevice);
		cuda_error_parser(__FUNCSIG__, "cudaMemcpy", cuda_err, "other.mat_ -> mat_");
#else
		memcpy(mat_, other.Get(), kSize);
#endif
		
		return *this;
	}

	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M>& DMatrix<N, M>::operator=(matrix_init_list_t &matrix)
	{
		*this = DMatrix(matrix);
		return *this;
	}

	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M> &DMatrix<N, M>::operator-()
	{
		const float kMinusOne = -1.0F;
		const cublasStatus_t cublas_err = cublasSscal(cublas_handle_, kLength, &kMinusOne, mat_, 1);
		assert(CUBLAS_STATUS_SUCCESS == cublas_err);
		return *this;
	}

	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M> &DMatrix<N, M>::operator += (const DMatrix &rhs)
	{
		const float kUnity = 1.0F;
		const cublasStatus_t cublas_err = cublasSaxpy(cublas_handle_, kLength, &kUnity, rhs.mat_, 1, mat_, 1);
		assert(CUBLAS_STATUS_SUCCESS == cublas_err);
		return *this;
	}
	
	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M> &DMatrix<N, M>::operator *= (const float d_rhs)
	{
		const cublasStatus_t cublas_err = cublasSscal(cublas_handle_, kLength, &d_rhs, mat_, 1);
		assert(CUBLAS_STATUS_SUCCESS == cublas_err);
		return *this;
	}

	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M> &DMatrix<N, M>::operator *= (const float* d_rhs)
	{
		const cublasStatus_t cublas_err = cublasSscal(cublas_handle_, kLength, d_rhs, mat_, 1);
		assert(CUBLAS_STATUS_SUCCESS == cublas_err);
		return *this;
	}
	
	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M> operator+(DMatrix<N, M> lhs, const DMatrix<N, M> &rhs)
	{
		lhs += rhs;
		return lhs;
	}

	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M> operator+(DMatrix<N, M> lhs, const DMatrix<N, 1>& rhs)
	{
		cublasStatus_t cublas_err = CUBLAS_STATUS_SUCCESS;
		const float kUnity = 1.0F;
		ptrdiff_t offset = 0;

		for (size_t i = 0; (i < lhs.kRowsN) && (CUBLAS_STATUS_SUCCESS == cublas_err); i++)
		{
			offset = i;
			cublas_err = cublasSaxpy(lhs.cublas_handle_, lhs.kLength - offset, &kUnity, rhs.Get(), 1, lhs.mat_ + offset, lhs.kRowsN);
		}
		assert(CUBLAS_STATUS_SUCCESS == cublas_err);

		return lhs;
	}

	template <size_t N, size_t M>
	__host__ __device__ DMatrix<N, M> operator+(DMatrix<N, M> lhs, const DMatrix<1, M>& rhs)
	{
		cublasStatus_t cublas_err = CUBLAS_STATUS_SUCCESS;
		const float kUnity = 1.0F;
		ptrdiff_t offset = 0;

		for (size_t i = 0; (i < lhs.kColsN) && (CUBLAS_STATUS_SUCCESS == cublas_err); i++)
		{
			offset = i * lhs.kRowsN;
			cublas_err = cublasSaxpy(lhs.cublas_handle_, lhs.kLength - offset, &kUnity, rhs.Get(), 1, lhs.mat_ + offset, 1);
		}

		assert(CUBLAS_STATUS_SUCCESS == cublas_err);

		return lhs;
	}

	template <size_t N, size_t M, size_t P>
	__host__ __device__ DMatrix<P, M> operator*(const DMatrix<N, M>& lhs, const DMatrix<P, N>& rhs)
	{
		const float kZero = 0.0F;
		const float kUnity = 1.0F;

		DMatrix<MATRIX_DOTP_DIM(lhs,rhs)> result;
		const cublasStatus_t cublas_err = cublasSgemm(
			lhs.cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
			result.kRowsN, result.kColsN, lhs.kColsN,
			&kUnity, lhs.mat_, lhs.kRowsN, rhs.mat_, rhs.kRowsN, &kZero, result.mat_, result.kRowsN
		);

		assert(CUBLAS_STATUS_SUCCESS == cublas_err);

		return result;
	}

	template <size_t N, size_t M>
	std::ostream& operator<<(std::ostream& os, const DMatrix<N, M>& m)
	{
		os << std::fixed << std::setprecision(2);
		os << "{";
		for (size_t row_i = 0; row_i < m.kRowsN; row_i++)
		{
			if (row_i != 0)
				os << "," << std::endl << "{ ";
			else
				os << "{";

			for (size_t col_i = 0; col_i < m.kColsN; col_i++)
			{
				if (col_i != 0)
					os << ",";
				os << std::setw(9) << std::scientific << m.At(col_i, row_i);
			}
			os << " }";
		}
		os << "}";

		return os;
	}

} // Namespace ai


#endif // MATRIX_CU_

/*
 *	--- End of file ---
 */
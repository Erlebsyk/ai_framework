/**
 * \file	matrix_test.cu
 * \brief	
 * 
 * \author	Erlend Isachsen (erlend.isachsen@7sense.no)
 * 
*/

/** Related header include	*/
#include "matrix.cuh"

/** Standard library header includes	*/
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <sstream>

/** Project header includes	*/

namespace utils
{
	/** Variable declarations	*/
	const float h_kMinusOne	= -1.0F;
	const float h_kZero		= 0.0F;
	const float h_kUnity	= 1.0F;

	/** Funciton declarations	*/
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

	__global__ void d_normalize_full();
	__global__ void d_normalize_by_row();
	__global__ void d_normalize_by_column();


	/** Classes, structs and enums	*/


	/** Funciton Implementations	 */

	void cuda_error_parser(
		const std::string caller_f,
		const std::string called_f,
		const cudaError_t error,
		const std::string context
	)
	{
		if (cudaSuccess != error)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << caller_f << "]";

			if (context != "")
				err_msg << " with context [" << context << "]. ";
			else
				err_msg << ". ";

			err_msg << "The function [" << called_f << "] returned with error code [" << error << "].";
			throw(std::runtime_error(err_msg.str()));
		}
	}

	void cublas_error_parser(
		std::string caller_f,
		std::string called_f,
		cublasStatus_t status,
		std::string context
	)
	{
		if (CUBLAS_STATUS_SUCCESS != status)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << caller_f << "]";

			if (context != "")
				err_msg << " with context [" << context << "]. ";
			else
				err_msg << ". ";

			err_msg << "The function [" << called_f << "] returned with status code [" << status << "].";
			throw(std::runtime_error(err_msg.str()));
		}
	}

	__global__ void d_normalize_full()
	{

	}

	__global__ void d_normalize_by_row()
	{

	}

	__global__ void d_normalize_by_column()
	{

	}

	Matrix::Matrix(const uint32_t n_cols, const uint32_t n_rows, float* mat) : 
		n_cols_	{ n_cols	},
		n_rows_	{ n_rows 	},
		mat_	{ mat		}
	{

	}

	Matrix::Matrix(const uint32_t n_cols, const uint32_t n_rows) :
		Matrix( n_cols, n_rows, (float*)malloc(sizeof(float) * n_cols * n_rows) )
	{

	}

	Matrix::Matrix(const Matrix &other) : 
		Matrix( other.n_cols_, other.n_rows_ )
	{
		memcpy(mat_, other.mat_, Size());
	}

	Matrix::Matrix(const DMatrix &d_mat) : 
		Matrix(d_mat.n_cols_, d_mat.n_rows_)
	{
		const cudaError_t cuda_err = cudaMemcpy(mat_, d_mat.mat_, Size(), cudaMemcpyDeviceToHost);
		cuda_error_parser(__PRETTY_FUNCTION__, "cudaMemcpy", cuda_err, "d_mat.mat_ -> mat_");
	}

	Matrix::~Matrix()
	{
		if(mat_ != nullptr)
		{
			free(mat_);
			mat_ = nullptr;
		}
	}

	Matrix::Matrix(Matrix &&other) : 
		n_cols_	{ other.n_cols_ },
		n_rows_	{ other.n_rows_ },
		mat_	{ other.mat_ }
	{
		other.mat_ = nullptr;
		other.~Matrix();
	}

	const float *Matrix::Get() const
	{
		return mat_;
	}

	float Matrix::At(const uint32_t col_i, const uint32_t row_i) const
	{
		const size_t index = (size_t)n_rows_ * col_i + row_i;
		return mat_[index];
	}

	void Matrix::Set(const uint32_t col_i, const uint32_t row_i, const float val)
	{
		const size_t index = (size_t)n_rows_ * col_i + row_i;
		mat_[index] = val;
	}

	size_t Matrix::Size() const
	{
		return sizeof(float) * Length();
	}

	size_t Matrix::Length() const
	{
		return n_cols_ * n_rows_;
	}

	Matrix& Matrix::operator=(const Matrix& other)
	{
		if((n_cols_ != other.n_cols_) || (n_rows_ != other.n_rows_))
		{
			/** \todo throw eception */
		}

		memcpy( mat_, other.mat_, Size() );
		return *this;
	}

	Matrix& Matrix::operator=(const DMatrix& other)
	{
		if((n_cols_ != other.n_cols_) || (n_rows_ != other.n_rows_))
		{
			/** \todo throw eception */
		}

		const cudaError_t cuda_err = cudaMemcpy(mat_, other.mat_, Size(), cudaMemcpyDeviceToHost);
		cuda_error_parser(__PRETTY_FUNCTION__, "cudaMemcpy", cuda_err, "other.mat_ -> mat_");
		return *this;
	}

	bool Matrix::operator==(const Matrix& other) const
	{
		bool equal = true;
		if(n_cols_ != other.n_cols_)
			equal = false;
		else if(n_rows_ != other.n_rows_)
			equal = false;
		
		for(size_t col_i = 0; (col_i < n_cols_) && (equal); col_i++)
		{
			for(size_t row_i = 0; (row_i < n_rows_) && (equal); row_i++)
			{
				equal = fabsf(At(col_i, row_i) - other.At(col_i, row_i)) < FLT_EPSILON;
				equal = equal || (isnan(At(col_i, row_i) ) && other.At(col_i, row_i));
			}
		}

		return equal;
	}

	std::ostream& operator<<(std::ostream& os, const Matrix& m)
	{
		os << std::fixed << std::setprecision(2);
		os << "{";
		for(uint32_t row_i = 0; row_i < m.n_rows_; row_i++)
		{
			if(row_i != 0)
				os << "," << std::endl << "{ ";
			else
				os << "{";

			for(uint32_t col_i = 0; col_i < m.n_cols_; col_i++)
			{
				if(col_i != 0)
					os << ",";
				os << std::setw(9) << std::scientific << m.At(col_i, row_i);
			}
			os << " }";
		}
		os << "}";

		return os;
	}

	DMatrix::DMatrix(const uint32_t n_cols, const uint32_t n_rows) : 
		Matrix(n_cols, n_rows, nullptr),
		d_n_cols_ { nullptr },
		d_n_rows_ { nullptr },
		cublas_handle_{}
	{
		// Allocate the matrix buffer on the device.
		cudaError_t cuda_err = cudaMalloc(&mat_, Size());
		cuda_error_parser(__PRETTY_FUNCTION__, "cudaMalloc", cuda_err, "mat_");

		cuda_err = cudaMalloc(&d_n_cols_, sizeof(uint32_t));
		cuda_error_parser(__PRETTY_FUNCTION__, "cudaMalloc", cuda_err, "d_n_cols_");

		cuda_err = cudaMalloc(&d_n_rows_, sizeof(uint32_t));
		cuda_error_parser(__PRETTY_FUNCTION__, "cudaMalloc", cuda_err, "d_n_rows_");

		cuda_err = cudaMemcpy(d_n_cols_, &n_cols_, sizeof(uint32_t), cudaMemcpyHostToDevice);
		cuda_error_parser(__PRETTY_FUNCTION__, "cudaMemcpy", cuda_err, "n_cols_ -> d_n_cols_");

		cuda_err = cudaMemcpy(d_n_rows_, &n_rows_, sizeof(uint32_t), cudaMemcpyHostToDevice);
		cuda_error_parser(__PRETTY_FUNCTION__, "cudaMemcpy", cuda_err, "n_rows_ -> d_n_rows_");

		// Create the cublas handle structure
		const cublasStatus_t cublas_err = cublasCreate(&cublas_handle_);
		cublas_error_parser(__PRETTY_FUNCTION__, "cublasCreate", cublas_err, "cublas_handle_");
	}

	DMatrix::DMatrix(const DMatrix &other) : 
		DMatrix(other.n_cols_, other.n_rows_)
	{
		const cudaError_t cuda_err = cudaMemcpy(mat_, other.mat_, Size(), cudaMemcpyDeviceToDevice);
		cuda_error_parser(__PRETTY_FUNCTION__, "cudaMemcpy", cuda_err, "other.mat_ -> mat_");
	}

	DMatrix::DMatrix(DMatrix &&other) : 
		Matrix(other.n_cols_, other.n_rows_, other.mat_),
		d_n_cols_		{ other.d_n_cols_		},
		d_n_rows_		{ other.d_n_rows_		},
		cublas_handle_	{ other.cublas_handle_	}
	{
		other.mat_ = nullptr;
		other.d_n_cols_ = nullptr;
		other.d_n_rows_ = nullptr;
		other.cublas_handle_ = nullptr;
	}

	DMatrix::DMatrix(const Matrix &h_mat) : 
		DMatrix(h_mat.n_cols_, h_mat.n_rows_)
	{
		const cudaError_t cuda_err = cudaMemcpy(mat_, h_mat.Get(), Size(), cudaMemcpyHostToDevice);
		cuda_error_parser(__PRETTY_FUNCTION__, "cudaMemcpy", cuda_err, "h_mat -> mat_");
	}

	DMatrix::DMatrix(const std::vector<std::vector<float>> &mat) :
		DMatrix( mat.front().size(), mat.size() )
	{
		for (size_t row_i = 0; row_i < mat.size(); row_i++)
		{
			if (mat.at(row_i).size() != n_cols_)
			{
				std::stringstream err_msg;
				err_msg << "Dimension error in [" << __PRETTY_FUNCTION__ << "] while constructing matrix from vector. "
						<< "All rows must be of equal size.";
				throw(std::runtime_error(err_msg.str()));
			}
		}

		for (size_t row_i = 0; row_i < n_rows_; row_i++)
		{
			for (size_t col_i = 0; col_i < n_cols_; col_i++)
			{
				Set(col_i, row_i, mat.at(row_i).at(col_i));
			}
		}
	}

	DMatrix::~DMatrix()
	{
		if(mat_ != nullptr)
		{
			cudaFree(mat_);
			mat_ = nullptr;
		}
		if (d_n_cols_ != nullptr)
		{
			cudaFree(d_n_cols_);
			d_n_cols_ = nullptr;
		}
		if (d_n_rows_ != nullptr)
		{
			cudaFree(d_n_rows_);
			d_n_rows_ = nullptr;
		}
		if(cublas_handle_ != nullptr)
		{
			cublasDestroy(cublas_handle_);
			cublas_handle_ = nullptr;
		}
	}

	float DMatrix::At(const uint32_t col_i, const uint32_t row_i) const
	{
		float h_val;
		const cudaError_t cuda_err = cudaMemcpy(&h_val, &(mat_[n_rows_ * col_i + row_i]), sizeof(float), cudaMemcpyDeviceToHost);
		cuda_error_parser(__PRETTY_FUNCTION__, "cudaMemcpy", cuda_err, "mat_[n_rows_ * col_i + row_i] -> h_val");
		return h_val;
	}

	void DMatrix::Set(const uint32_t col_i, const uint32_t row_i, const float val)
	{
		const cudaError_t cuda_err = cudaMemcpy(&(mat_[n_rows_ * col_i + row_i]), &val, sizeof(float), cudaMemcpyHostToDevice);
		cuda_error_parser(__PRETTY_FUNCTION__, "cudaMemcpy", cuda_err, "val -> mat_[n_rows_ * col_i + row_i]");
	}

	__device__ float DMatrix::d_At(const uint32_t col_i, const uint32_t row_i) const
	{
		return mat_[n_rows_ * col_i + row_i];
	}

	__device__ void DMatrix::d_Set(const uint32_t col_i, const uint32_t row_i, const float val)
	{
		mat_[n_rows_ * col_i + row_i] = val;
	}

	float DMatrix::SumGet() const
	{
		float sum;
		const cublasStatus_t cublas_err = cublasSasum(cublas_handle_, Length(), Get(), 1, &sum);
		cublas_error_parser(__PRETTY_FUNCTION__, "cublasSasum", cublas_err);
		return sum;
	}

	float DMatrix::SumRowGet(const size_t row_i) const
	{
		// Boundary check
		if (row_i >= n_rows_)
		{
			std::stringstream err_msg;
			err_msg << "Out of range error in [" << __PRETTY_FUNCTION__ << "]. "
				<< "The requested row was [" << row_i << "], but the matrix only has [" << n_rows_ << "] rows.";
			throw(std::runtime_error(err_msg.str()));
		}

		// Obtain the sum
		float row_sum;
		const cublasStatus_t cublas_err = cublasSasum(cublas_handle_, n_cols_, Get() + row_i, n_rows_, &row_sum);
		cublas_error_parser(__PRETTY_FUNCTION__, "cublasSasum", cublas_err);

		return row_sum;
	}

	float DMatrix::SumColGet(const size_t col_i) const
	{
		// Boundary check
		if (col_i >= n_cols_)
		{
			std::stringstream err_msg;
			err_msg << "Out of range error in [" << __PRETTY_FUNCTION__ << "]. "
				<< "The requested column was [" << col_i << "], but the matrix only has [" << n_cols_ << "] columns.";
			throw(std::runtime_error(err_msg.str()));
		}

		// Obtain the sum
		float col_sum;
		const cublasStatus_t cublas_err = cublasSasum(cublas_handle_, n_rows_, Get() + col_i * n_rows_, 1, &col_sum);
		cublas_error_parser(__PRETTY_FUNCTION__, "cublasSasum", cublas_err);

		return col_sum;
	}

	void DMatrix::NormalizeFull()
	{
		*this *= (1.0f / SumGet());
	}

	void DMatrix::NormalizeByRow()
	{
		float row_factor;
		cublasStatus_t cublas_err;
		for (size_t i = 0; i < n_rows_; i++)
		{
			row_factor = 1.0f / SumRowGet(i);
			cublas_err = cublasSscal(cublas_handle_, n_cols_, &row_factor, mat_ + i, n_rows_);
			cublas_error_parser(__PRETTY_FUNCTION__, "cublasSscal", cublas_err);
		}
	}

	void DMatrix::NormalizeByColumn()
	{
		float col_factor;
		cublasStatus_t cublas_err;
		for (size_t i = 0; i < n_cols_; i++)
		{
			col_factor = 1.0f / SumColGet(i);
			cublas_err = cublasSscal(cublas_handle_, n_rows_, &col_factor, mat_ + i * n_rows_, 1);
			cublas_error_parser(__PRETTY_FUNCTION__, "cublasSscal", cublas_err);
		}
	}


	DMatrix& DMatrix::operator=(const DMatrix& other)
	{
		if((n_cols_ != other.n_cols_) || (n_rows_ != other.n_rows_))
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "]. Incompatible sizes. The two matrices must have equal dimensions.";
			throw(std::runtime_error(err_msg.str()));
		}
		const cudaError_t cuda_err = cudaMemcpy(mat_, other.Get(), Size(), cudaMemcpyDeviceToDevice);
		cuda_error_parser(__PRETTY_FUNCTION__, "cudaMemcpy", cuda_err, "other.mat_ -> mat_");
		return *this;
	}

	DMatrix& DMatrix::operator=(const Matrix& other)
	{
		if((n_cols_ != other.n_cols_) || (n_rows_ != other.n_rows_))
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "]."
					<< "The two matrices have incompatible sizes.The two matrices must have equal dimensions.";
			throw(std::runtime_error(err_msg.str()));
		}
		const cudaError_t cuda_err = cudaMemcpy(mat_, other.Get(), Size(), cudaMemcpyHostToDevice);
		cuda_error_parser(__PRETTY_FUNCTION__, "cudaMemcpy", cuda_err, "other.mat_ -> mat_");
		return *this;
	}

	DMatrix& DMatrix::operator=(const std::vector<std::vector<float>> &matrix)
	{
		*this = DMatrix(matrix);
		return *this;
	}

	DMatrix &DMatrix::operator-()
	{
		const cublasStatus_t cublas_err = cublasSscal(cublas_handle_, Length(), &h_kMinusOne, mat_, 1);
		cublas_error_parser(__PRETTY_FUNCTION__, "cublasSscal", cublas_err);
		return *this;
	}


	DMatrix& DMatrix::operator += (const DMatrix& rhs)
	{
		cublasStatus_t cublas_err = CUBLAS_STATUS_SUCCESS;
		if ((n_cols_ == rhs.n_cols_) && (n_rows_ == rhs.n_rows_))
		{
			cublas_err = cublasSaxpy(cublas_handle_, Length(), &h_kUnity, rhs.mat_, 1, mat_, 1);
		}
		else if ((n_cols_ == rhs.n_cols_) && (1 == rhs.n_rows_))
		{
			ptrdiff_t offset = 0;
			for (size_t i = 0; (i < n_rows_) && (CUBLAS_STATUS_SUCCESS == cublas_err); i++)
			{
				offset = i;
				cublas_err = cublasSaxpy(cublas_handle_, Length() - offset, &h_kUnity, rhs.mat_, 1, mat_ + offset, n_rows_);
			}
		}
		else if ((n_rows_ == rhs.n_rows_) && (1 == rhs.n_cols_))
		{
			ptrdiff_t offset = 0;
			for (size_t i = 0; (i < n_cols_) && (CUBLAS_STATUS_SUCCESS == cublas_err); i++)
			{
				offset = i * n_rows_;
				cublas_err = cublasSaxpy(cublas_handle_, Length() - offset, &h_kUnity, rhs.mat_, 1, mat_ + offset, 1);
			}
		}
		else
		{
			cublas_err = CUBLAS_STATUS_NOT_SUPPORTED;
		}
		cublas_error_parser(__PRETTY_FUNCTION__, "cublasSaxpy", cublas_err);
		
		return *this;
	}
	
	DMatrix& DMatrix::operator *= (const float d_rhs)
	{
		const cublasStatus_t cublas_err = cublasSscal(cublas_handle_, Length(), &d_rhs, mat_, 1);
		cublas_error_parser(__PRETTY_FUNCTION__, "cublasSscal", cublas_err);
		return *this;
	}

	DMatrix& DMatrix::operator *= (const float* d_rhs)
	{
		const cublasStatus_t cublas_err = cublasSscal(cublas_handle_, Length(), d_rhs, mat_, 1);
		cublas_error_parser(__PRETTY_FUNCTION__, "cublasSscal", cublas_err);
		return *this;
	}
		
	DMatrix operator+(DMatrix lhs, const DMatrix& rhs)
	{
		lhs += rhs;
		return lhs;
	}

	DMatrix operator*(const DMatrix &lhs, const DMatrix& rhs)
	{
		if(lhs.n_cols_ != rhs.n_rows_)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "]. The two matrices have incompatible size. "
				<< "The number of columns in left hand side must equal the number of rows in right hand side. "
				<< "The matrices are: " << std::endl << lhs << "," << std::endl << rhs;
			throw(std::runtime_error(err_msg.str()));
		}

		DMatrix result(rhs.n_cols_, lhs.n_rows_);

		const cublasStatus_t cublas_err = cublasSgemm(
			lhs.cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
			result.n_rows_, result.n_cols_, lhs.n_cols_,
			&h_kUnity, lhs.mat_, lhs.n_rows_, rhs.mat_, rhs.n_rows_, &h_kZero, result.mat_, result.n_rows_
		);
		cublas_error_parser(__PRETTY_FUNCTION__, "cublasSgemm", cublas_err);

		return result;
	}


} // Namespace ai


/*
 *	--- End of file ---
 */
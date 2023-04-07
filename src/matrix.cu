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
	void test()
	{
		std::cout << "In test" << std::endl;
	}

	/** Classes, structs and enums	*/


	/** Funciton Implementations	 */
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
		cudaMemcpy(mat_, d_mat.mat_, Size(), cudaMemcpyDeviceToHost);
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
		if((this->n_cols_ != other.n_cols_) || (this->n_rows_ != other.n_rows_))
		{
			/** \todo throw eception */
		}

		memcpy( this->mat_, other.mat_, Size() );
		return *this;
	}

	Matrix& Matrix::operator=(const DMatrix& other)
	{
		if((this->n_cols_ != other.n_cols_) || (this->n_rows_ != other.n_rows_))
		{
			/** \todo throw eception */
		}

		cudaMemcpy(this->mat_, other.mat_, Size(), cudaMemcpyDeviceToHost);
		return *this;
	}

	bool Matrix::operator==(const Matrix& other) const
	{
		bool equal = true;
		if(this->n_cols_ != other.n_cols_)
			equal = false;
		else if(this->n_rows_ != other.n_rows_)
			equal = false;
		
		for(size_t col_i = 0; (col_i < n_cols_) && (equal); col_i++)
		{
			for(size_t row_i = 0; (row_i < n_rows_) && (equal); row_i++)
			{
				equal = fabsf(this->At(col_i, row_i) - other.At(col_i, row_i)) < FLT_EPSILON;
				equal = equal || (isnan(this->At(col_i, row_i) ) && other.At(col_i, row_i));
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
		cublas_handle_{}
	{
		// Allocate the matrix buffer on the device.
		const cudaError_t cuda_err = cudaMalloc(&mat_, Size());
		if (cudaSuccess != cuda_err)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "]. "
					<< "The function 'cudaMalloc' returned with error code [" << cuda_err << "].";
			throw(std::runtime_error(err_msg.str()));
		}
		
		// Create the cublas handle structure
		const cublasStatus_t cublas_err = cublasCreate_v2(&cublas_handle_);
		if (CUBLAS_STATUS_SUCCESS != cublas_err)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "]. "
					<< "The function 'cublasCreate_v2' returned with error code [" << cublas_err << "].";
			throw( std::runtime_error(err_msg.str()) );
		}
	}

	DMatrix::DMatrix(const DMatrix &other) : 
		DMatrix(other.n_cols_, other.n_rows_)
	{
		const cudaError_t cuda_err = cudaMemcpy(mat_, other.mat_, Size(), cudaMemcpyDeviceToDevice);
		if (cudaSuccess != cuda_err)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "] while attempting to copy data from device to device. "
					<< "The function 'cudaMemcpy' returned with error code [" << cuda_err << "].";
			throw(std::runtime_error(err_msg.str()));
		}
	}

	DMatrix::DMatrix(DMatrix &&other) : 
		Matrix(other.n_cols_, other.n_rows_, other.mat_),
		cublas_handle_{ other.cublas_handle_ }
	{
		other.mat_ = nullptr;
		other.cublas_handle_ = nullptr;
	}

	DMatrix::DMatrix(const Matrix &h_mat) : 
		DMatrix(h_mat.n_cols_, h_mat.n_rows_)
	{
		const cudaError_t cuda_err = cudaMemcpy(mat_, h_mat.Get(), Size(), cudaMemcpyHostToDevice);
		if (cudaSuccess != cuda_err)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "] while attempting to copy data from host to device. "
					<< "The function 'cudaMemcpy' returned with error code [" << cuda_err << "].";
			throw(std::runtime_error(err_msg.str()));
		}
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
		if(cublas_handle_ != nullptr)
		{
			cublasDestroy_v2(cublas_handle_);
			cublas_handle_ = nullptr;
		}
	}

	float DMatrix::At(const uint32_t col_i, uint32_t row_i) const
	{
		float h_val;
		const cudaError_t cuda_err = cudaMemcpy(&h_val, &(mat_[n_rows_ * col_i + row_i]), sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaSuccess != cuda_err)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "] while attempting to copy data from device to host. "
					<< "The function 'cudaMemcpy' returned with error code [" << cuda_err << "].";
			throw(std::runtime_error(err_msg.str()));
		}
		return h_val;
	}

	void DMatrix::Set(const uint32_t col_i, const uint32_t row_i, const float val)
	{
		const cudaError_t cuda_err = cudaMemcpy(&(mat_[n_rows_ * col_i + row_i]), &val, sizeof(float), cudaMemcpyHostToDevice);
		if (cudaSuccess != cuda_err)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "] while attempting to copy data from host to device. "
					<< "The function 'cudaMemcpy' returned with error code [" << cuda_err << "].";
			throw(std::runtime_error(err_msg.str()));
		}
	}

	DMatrix& DMatrix::operator=(const DMatrix& other)
	{
		if((this->n_cols_ != other.n_cols_) || (this->n_rows_ != other.n_rows_))
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "]. Incompatible sizes. The two matrices must have equal dimensions.";
			throw(std::runtime_error(err_msg.str()));
		}
		cudaMemcpy(mat_, other.Get(), Size(), cudaMemcpyDeviceToDevice);
		return *this;
	}

	DMatrix& DMatrix::operator=(const Matrix& other)
	{
		if((this->n_cols_ != other.n_cols_) || (this->n_rows_ != other.n_rows_))
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "]."
					<< "The two matrices have incompatible sizes.The two matrices must have equal dimensions.";
			throw(std::runtime_error(err_msg.str()));
		}
		cudaMemcpy(mat_, other.Get(), Size(), cudaMemcpyHostToDevice);
		return *this;
	}

	DMatrix& DMatrix::operator=(const std::vector<std::vector<float>> &matrix)
	{
		*this = DMatrix(matrix);
		return *this;
	}

	DMatrix &DMatrix::operator-()
	{
		const cublasStatus_t cublas_err = cublasSscal_v2(cublas_handle_, Length(), &h_kMinusOne, mat_, 1);
		if (CUBLAS_STATUS_SUCCESS != cublas_err)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "]. "
					<< "The function 'cublasSscal_v2' returned with error code [" << cublas_err << "].";
			throw(std::runtime_error(err_msg.str()));
		}
		return *this;
	}


	DMatrix& DMatrix::operator += (const DMatrix& rhs)
	{
		const cublasStatus_t cublas_err = cublasSaxpy_v2(cublas_handle_, Length(), &h_kUnity, rhs.mat_, 1, this->mat_, 1);
		if (CUBLAS_STATUS_SUCCESS != cublas_err)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "]. "
				<< "The function 'cublasSaxpy_v2' returned with error code [" << cublas_err << "].";
			throw(std::runtime_error(err_msg.str()));
		}
		return *this;
	}
	
	DMatrix& DMatrix::operator *= (const float d_rhs)
	{
		const cublasStatus_t cublas_err = cublasSscal_v2(cublas_handle_, Size(), &d_rhs, mat_, 1);
		if (CUBLAS_STATUS_SUCCESS != cublas_err)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "]. "
				<< "The function 'cublasSscal_v2' returned with error code [" << cublas_err << "].";
			throw(std::runtime_error(err_msg.str()));
		}
		return *this;
	}

	DMatrix& DMatrix::operator *= (const float* d_rhs)
	{
		const cublasStatus_t cublas_err = cublasSscal_v2(cublas_handle_, Size(), d_rhs, mat_, 1);
		if (CUBLAS_STATUS_SUCCESS != cublas_err)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "]. "
				<< "The function 'cublasSscal_v2' returned with error code [" << cublas_err << "].";
			throw(std::runtime_error(err_msg.str()));
		}
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

		const cublasStatus_t cublas_err = cublasSgemm_v2(
			lhs.cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
			result.n_rows_, result.n_cols_, lhs.n_cols_,
			&h_kUnity, lhs.mat_, lhs.n_rows_, rhs.mat_, rhs.n_rows_, &h_kZero, result.mat_, result.n_rows_
		);

		if (CUBLAS_STATUS_SUCCESS != cublas_err)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << __PRETTY_FUNCTION__ << "]. "
				<< "The function 'cublasSgemm_v2' returned with error code [" << cublas_err << "].";
			throw(std::runtime_error(err_msg.str()));
		}

		return result;
	}


} // Namespace ai


/*
 *	--- End of file ---
 */
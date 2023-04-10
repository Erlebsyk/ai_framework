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
#include <vector>

/** NRF / Zephyr header includes	*/


/** Project header includes	*/


namespace utils
{
	/** Variable declarations	*/
	class DMatrix;
	class Matrix;

	/** Funciton declarations	*/

	/** Classes, structs and enums	*/
	class Matrix
	{
	public:
		const uint32_t n_cols_;
		const uint32_t n_rows_;

	protected:
		float *mat_;
		Matrix(uint32_t n_cols, uint32_t n_rows, float* mat);

	public:
		Matrix(uint32_t n_cols, uint32_t n_rows);
		Matrix(const Matrix &other);
		Matrix(Matrix &&other);
		Matrix(const DMatrix &d_mat);
		~Matrix();

		virtual const float *Get() const;
		virtual float At(uint32_t col_i, uint32_t row_i) const;
		virtual void Set(uint32_t col_i, uint32_t row_i, float val);

		size_t Size() const;
		size_t Length() const;

		Matrix &operator=(const Matrix& other);
		Matrix &operator=(const DMatrix& other);

		bool operator==(const Matrix& other) const;

		friend std::ostream &operator<<(std::ostream& os, const Matrix& m);
	};

	class DMatrix : public Matrix
	{
	protected:
		cublasHandle_t cublas_handle_;

	public:
		DMatrix(uint32_t n_cols, uint32_t n_rows);
		DMatrix(const DMatrix &other);
		DMatrix(DMatrix &&other);
		DMatrix(const Matrix &h_mat);
		DMatrix(const std::vector<std::vector<float>> &mat);
		~DMatrix();

		float At(uint32_t col_i, uint32_t row_i) const;
		void Set(uint32_t col_i, uint32_t row_i, float val);

		float SumGet() const;
		float SumRowGet(size_t row_i) const;
		float SumColGet(size_t col_i) const;

		void NormalizeFull();
		void NormalizeByRow();
		void NormalizeByColumn();

		DMatrix &operator=(const DMatrix& other);
		DMatrix &operator=(const Matrix& other);
		DMatrix &operator=(const std::vector<std::vector<float>> &matrix);
		DMatrix &operator-();
		
		//DMatrix& operator += (const float* d_rhs);
		DMatrix &operator += (const DMatrix& rhs);
		
		DMatrix &operator *= (const float* d_rhs);
		DMatrix &operator *= (float d_rhs);
			
		friend DMatrix operator+(DMatrix lhs, const DMatrix& rhs);
		friend DMatrix operator*(const DMatrix& lhs, const DMatrix& rhs);
	};

	/** Funciton Implementations	 */


} // Namespace ai

#endif // MATRIX_CUH_

/*
 *	--- End of file ---
 */
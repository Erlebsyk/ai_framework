/**
 * \file	matrix_test.cpp
 * \brief	File for matrix tests.
 * 
 * \author	Erlend Isachsen (erlend.isachsen@7sense.no)
 * \copyright	Copyright (c) 2023 7Sense technologies
 * 
*/

/** Standard library header includes	*/
#include <cstdint>
#include <gtest/gtest.h>

/** Project header includes	*/
#include "../src/matrix.cuh"

namespace test
{
	/** Variable declarations	*/


	/** Funciton declarations	*/


	/** Classes, structs and enums	*/


	/** Funciton Implementations	 */
	TEST(matrix_test, construction)
	{
		utils::Matrix A(3,2);
		EXPECT_EQ(A.n_cols_, 3);
		EXPECT_EQ(A.n_rows_, 2);
		EXPECT_EQ(A.Size(), sizeof(float)*6);

		utils::Matrix B(4,5);
		EXPECT_EQ(B.n_cols_, 4);
		EXPECT_EQ(B.n_rows_, 5);
		EXPECT_EQ(B.Size(), sizeof(float)*20);

		utils::Matrix C(10,10);
		EXPECT_EQ(C.n_cols_, 10);
		EXPECT_EQ(C.n_rows_, 10);
		EXPECT_EQ(C.Size(), sizeof(float)*100);

		utils::Matrix D = A;
		EXPECT_EQ(D.n_cols_, 3);
		EXPECT_EQ(D.n_rows_, 2);
		EXPECT_EQ(D.Size(), sizeof(float)*6);
		EXPECT_TRUE(D == A);

		utils::Matrix E(B);
		EXPECT_EQ(E.n_cols_, 4);
		EXPECT_EQ(E.n_rows_, 5);
		EXPECT_EQ(E.Size(), sizeof(float)*20);
		EXPECT_TRUE(E == B);

		utils::DMatrix dMat(4,4);
		utils::Matrix F = dMat;
		EXPECT_EQ(F.n_cols_, 4);
		EXPECT_EQ(F.n_rows_, 4);
		EXPECT_EQ(F.Size(), sizeof(float)*16);

		EXPECT_TRUE(F == dMat);

		std::vector<std::vector<float>> vec = {
			{1.0f, 2.0f, 3.0f},
			{ 4.0f, 5.0f, 6.0f },
			{ 7.0f, 8.0f, 9.0f }
		};
		utils::DMatrix G = vec;

		for (size_t row_i = 0; row_i < vec.size(); row_i++)
		{
			for (size_t col_i = 0; col_i < vec.front().size(); col_i++)
			{
				EXPECT_FLOAT_EQ(G.At(col_i, row_i), vec.at(row_i).at(col_i));
			}
		}

		vec.front().push_back(3.5f);
		EXPECT_THROW(G = vec, std::runtime_error);
	}

	TEST(matrix_test, access)
	{
		utils::Matrix A(3,2);
		for(size_t col_i = 0; col_i < A.n_cols_; col_i++)
		{
			for(size_t row_i = 0; row_i < A.n_rows_; row_i++)
			{
				A.Set(col_i, row_i, col_i * 10 + row_i);
			}
		}

		EXPECT_FLOAT_EQ(A.At(0,0), 0.0f);
		EXPECT_FLOAT_EQ(A.At(1,0), 10.0f);
		EXPECT_FLOAT_EQ(A.At(2,0), 20.0f);
		EXPECT_FLOAT_EQ(A.At(0,1), 1.0f);
		EXPECT_FLOAT_EQ(A.At(1,1), 11.0f);
		EXPECT_FLOAT_EQ(A.At(2,1), 21.0f);

		utils::Matrix B(2,3);
		for(size_t col_i = 0; col_i < B.n_cols_; col_i++)
		{
			for(size_t row_i = 0; row_i < B.n_rows_; row_i++)
			{
				B.Set(col_i, row_i, col_i * 10 + row_i);
			}
		}

		EXPECT_FLOAT_EQ(B.At(0,0), 0.0f);
		EXPECT_FLOAT_EQ(B.At(1,0), 10.0f);
		EXPECT_FLOAT_EQ(B.At(0,1), 1.0f);
		EXPECT_FLOAT_EQ(B.At(1,1), 11.0f);
		EXPECT_FLOAT_EQ(B.At(0,2), 2.0f);
		EXPECT_FLOAT_EQ(B.At(1,2), 12.0f);

		utils::Matrix C = B;
		EXPECT_TRUE( C == B);
		C.Set(1,1, C.At(1,1) + 0.0001f);
		EXPECT_FALSE( C == B );
	}

	TEST(device_matrix_test, construction)
	{
		utils::DMatrix A(3,2);
		EXPECT_EQ(A.n_cols_, 3);
		EXPECT_EQ(A.n_rows_, 2);
		EXPECT_EQ(A.Size(), sizeof(float)*6);

		utils::DMatrix B(4,5);
		EXPECT_EQ(B.n_cols_, 4);
		EXPECT_EQ(B.n_rows_, 5);
		EXPECT_EQ(B.Size(), sizeof(float)*20);

		utils::DMatrix C(10,10);
		EXPECT_EQ(C.n_cols_, 10);
		EXPECT_EQ(C.n_rows_, 10);
		EXPECT_EQ(C.Size(), sizeof(float)*100);

		utils::DMatrix D = A;
		EXPECT_EQ(D.n_cols_, 3);
		EXPECT_EQ(D.n_rows_, 2);
		EXPECT_EQ(D.Size(), sizeof(float)*6);
		EXPECT_TRUE(D == A);

		utils::DMatrix E(B);
		EXPECT_EQ(E.n_cols_, 4);
		EXPECT_EQ(E.n_rows_, 5);
		EXPECT_EQ(E.Size(), sizeof(float)*20);
		EXPECT_TRUE(E == B);

		utils::Matrix hMat(4,4);
		utils::DMatrix F = hMat;
		EXPECT_EQ(F.n_cols_, 4);
		EXPECT_EQ(F.n_rows_, 4);
		EXPECT_EQ(F.Size(), sizeof(float)*16);

		EXPECT_TRUE(F == hMat);
	}

	TEST(device_matrix_test, access)
	{
		utils::DMatrix A(3,2);
		for(size_t col_i = 0; col_i < A.n_cols_; col_i++)
		{
			for(size_t row_i = 0; row_i < A.n_rows_; row_i++)
			{
				A.Set(col_i, row_i, col_i * 10 + row_i);
			}
		}

		EXPECT_FLOAT_EQ(A.At(0,0), 0.0f);
		EXPECT_FLOAT_EQ(A.At(1,0), 10.0f);
		EXPECT_FLOAT_EQ(A.At(2,0), 20.0f);
		EXPECT_FLOAT_EQ(A.At(0,1), 1.0f);
		EXPECT_FLOAT_EQ(A.At(1,1), 11.0f);
		EXPECT_FLOAT_EQ(A.At(2,1), 21.0f);

		utils::DMatrix B(2,3);
		for(size_t col_i = 0; col_i < B.n_cols_; col_i++)
		{
			for(size_t row_i = 0; row_i < B.n_rows_; row_i++)
			{
				B.Set(col_i, row_i, col_i * 10 + row_i);
			}
		}

		EXPECT_FLOAT_EQ(B.At(0,0), 0.0f);
		EXPECT_FLOAT_EQ(B.At(1,0), 10.0f);
		EXPECT_FLOAT_EQ(B.At(0,1), 1.0f);
		EXPECT_FLOAT_EQ(B.At(1,1), 11.0f);
		EXPECT_FLOAT_EQ(B.At(0,2), 2.0f);
		EXPECT_FLOAT_EQ(B.At(1,2), 12.0f);

		utils::Matrix C = B;
		EXPECT_TRUE( C == B);
		C.Set(1,1, C.At(1,1) + 0.0001f);
		EXPECT_FALSE( C == B );
	}

	TEST(device_matrix_test, operators)
	{
		utils::DMatrix A(3,2);
		for(size_t col_i = 0; col_i < A.n_cols_; col_i++)
		{
			for(size_t row_i = 0; row_i < A.n_rows_; row_i++)
			{
				A.Set(col_i, row_i, col_i * 10 + row_i);
			}
		}

		utils::DMatrix ANeg = A;
		ANeg = -ANeg;
		for(size_t col_i = 0; col_i < ANeg.n_cols_; col_i++)
		{
			for(size_t row_i = 0; row_i < ANeg.n_rows_; row_i++)
			{
				EXPECT_FLOAT_EQ(ANeg.At(col_i, row_i), -A.At(col_i, row_i));
			}
		}

		utils::DMatrix B(3,2);
		for(size_t col_i = 0; col_i < B.n_cols_; col_i++)
		{
			for(size_t row_i = 0; row_i < B.n_rows_; row_i++)
			{
				B.Set(col_i, row_i, 2.0f);
			}
		}

		utils::DMatrix C(A);
		C += B;
		for(size_t col_i = 0; col_i < C.n_cols_; col_i++)
		{
			for(size_t row_i = 0; row_i < C.n_rows_; row_i++)
			{
				EXPECT_FLOAT_EQ(C.At(col_i, row_i), A.At(col_i, row_i) + 2);
			}
		}

		C = A + B;
		for(size_t col_i = 0; col_i < C.n_cols_; col_i++)
		{
			for(size_t row_i = 0; row_i < C.n_rows_; row_i++)
			{
				EXPECT_FLOAT_EQ(C.At(col_i, row_i), A.At(col_i, row_i) + 2);
			}
		}

		utils::DMatrix D(2,3);
		for(size_t col_i = 0; col_i < D.n_cols_; col_i++)
		{
			for(size_t row_i = 0; row_i < D.n_rows_; row_i++)
			{
				D.Set(col_i, row_i, col_i + row_i * D.n_cols_);
			}
		}

		D *= 2.0f;
		for(size_t col_i = 0; col_i < D.n_cols_; col_i++)
		{
			for(size_t row_i = 0; row_i < D.n_rows_; row_i++)
			{
				EXPECT_FLOAT_EQ(D.At(col_i, row_i), 2.0f * (col_i + row_i * D.n_cols_));
			}
		}

		utils::DMatrix E = C * D;
		utils::DMatrix F = D * C;

		const utils::DMatrix G = std::vector<std::vector<float>>{
			{ 224.0f,	296.0f	},
			{ 236.0f,	314.0f	}
		};
		const utils::DMatrix H = std::vector<std::vector<float>>{
			{ 6.0f,		26.0f,	46.0f	},
			{ 26.0f,	126.0f,	226.0f	},
			{ 46.0f,	226.0f,	406.0f	}
		};

		EXPECT_TRUE(E == G);
		EXPECT_TRUE(F == H);
	}

	
} // Namespace test


/*
 *	--- End of file ---
 */
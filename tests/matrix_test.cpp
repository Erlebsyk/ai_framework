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
#include <cuda_runtime.h>

/** Project header includes	*/
#define protected public
#include "../src/utils/matrix.cuh"

namespace test
{
	/** Variable declarations	*/


	/** Funciton declarations	*/


	/** Classes, structs and enums	*/


	/** Funciton Implementations	 */

	TEST(device_matrix_test, construction)
	{
		utils::DMatrix<3, 2> A;
		EXPECT_EQ(A.kColsN, 3);
		EXPECT_EQ(A.kRowsN, 2);
		EXPECT_EQ(A.kLength, 6);
		EXPECT_EQ(A.kSize, 6 * sizeof(float));


		utils::DMatrix<4, 5> B;
		EXPECT_EQ(B.kColsN, 4);
		EXPECT_EQ(B.kRowsN, 5);
		EXPECT_EQ(B.kLength, 20);
		EXPECT_EQ(B.kSize, 20 * sizeof(float));

		utils::DMatrix<10, 10> C;
		EXPECT_EQ(C.kColsN, 10);
		EXPECT_EQ(C.kRowsN, 10);
		EXPECT_EQ(C.kLength, 100);
		EXPECT_EQ(C.kSize, 100 * sizeof(float));

		utils::DMatrix<A.kColsN, A.kRowsN> D = A;
		EXPECT_EQ(D.kColsN, 3);
		EXPECT_EQ(D.kRowsN, 2);
		EXPECT_EQ(D.kLength, 6);
		EXPECT_EQ(D.kSize, 6 * sizeof(float));
		EXPECT_TRUE(D == A);

		utils::DMatrix<B.kColsN, B.kRowsN> E(B);
		EXPECT_EQ(E.kColsN, 4);
		EXPECT_EQ(E.kRowsN, 5);
		EXPECT_EQ(E.kSize, sizeof(float) * 20);
		EXPECT_TRUE(E == B);
	}

	TEST(device_matrix_test, access)
	{
		utils::DMatrix<3, 2> A;
		for (size_t col_i = 0; col_i < A.kColsN; col_i++)
		{
			for (size_t row_i = 0; row_i < A.kRowsN; row_i++)
			{
				A.Set(col_i, row_i, col_i * 10 + row_i);
			}
		}

		EXPECT_FLOAT_EQ(A.At(0, 0), 0.0f);
		EXPECT_FLOAT_EQ(A.At(1, 0), 10.0f);
		EXPECT_FLOAT_EQ(A.At(2, 0), 20.0f);
		EXPECT_FLOAT_EQ(A.At(0, 1), 1.0f);
		EXPECT_FLOAT_EQ(A.At(1, 1), 11.0f);
		EXPECT_FLOAT_EQ(A.At(2, 1), 21.0f);

		utils::DMatrix<2, 3> B;
		for (size_t col_i = 0; col_i < B.kColsN; col_i++)
		{
			for (size_t row_i = 0; row_i < B.kRowsN; row_i++)
			{
				B.Set(col_i, row_i, col_i * 10 + row_i);
			}
		}

		EXPECT_FLOAT_EQ(B.At(0, 0), 0.0f);
		EXPECT_FLOAT_EQ(B.At(1, 0), 10.0f);
		EXPECT_FLOAT_EQ(B.At(0, 1), 1.0f);
		EXPECT_FLOAT_EQ(B.At(1, 1), 11.0f);
		EXPECT_FLOAT_EQ(B.At(0, 2), 2.0f);
		EXPECT_FLOAT_EQ(B.At(1, 2), 12.0f);
	}

	TEST(device_matrix_test, operators)
	{
		utils::DMatrix<3, 2> A;
		for (size_t col_i = 0; col_i < A.kColsN; col_i++)
		{
			for (size_t row_i = 0; row_i < A.kRowsN; row_i++)
			{
				A.Set(col_i, row_i, col_i * 10 + row_i);
			}
		}

		utils::DMatrix<A.kColsN, A.kRowsN> ANeg = A;
		ANeg = -ANeg;
		for (size_t col_i = 0; col_i < ANeg.kColsN; col_i++)
		{
			for (size_t row_i = 0; row_i < ANeg.kRowsN; row_i++)
			{
				EXPECT_FLOAT_EQ(ANeg.At(col_i, row_i), -A.At(col_i, row_i));
			}
		}

		utils::DMatrix<3, 2> B;
		for (size_t col_i = 0; col_i < B.kColsN; col_i++)
		{
			for (size_t row_i = 0; row_i < B.kRowsN; row_i++)
			{
				B.Set(col_i, row_i, 2.0f);
			}
		}

		utils::DMatrix<A.kColsN, A.kRowsN> C(A);
		C += B;
		for (size_t col_i = 0; col_i < C.kColsN; col_i++)
		{
			for (size_t row_i = 0; row_i < C.kRowsN; row_i++)
			{
				EXPECT_FLOAT_EQ(C.At(col_i, row_i), A.At(col_i, row_i) + 2);
			}
		}

		C = A + B;
		for (size_t col_i = 0; col_i < C.kColsN; col_i++)
		{
			for (size_t row_i = 0; row_i < C.kRowsN; row_i++)
			{
				EXPECT_FLOAT_EQ(C.At(col_i, row_i), A.At(col_i, row_i) + 2);
			}
		}

		utils::DMatrix<2, 3> D;
		for (size_t col_i = 0; col_i < D.kColsN; col_i++)
		{
			for (size_t row_i = 0; row_i < D.kRowsN; row_i++)
			{
				D.Set(col_i, row_i, col_i + row_i * D.kColsN);
			}
		}

		D *= 2.0f;
		for (size_t col_i = 0; col_i < D.kColsN; col_i++)
		{
			for (size_t row_i = 0; row_i < D.kRowsN; row_i++)
			{
				EXPECT_FLOAT_EQ(D.At(col_i, row_i), 2.0f * (col_i + row_i * D.kColsN));
			}
		}

		utils::DMatrix<MATRIX_DOTP_DIM(C, D)> E = C * D;
		utils::DMatrix<MATRIX_DOTP_DIM(D, C)> F = D * C;

		const utils::DMatrix<2, 2> G = {
			{ 224.0f, 296.0f	},
			{ 236.0f, 314.0f	}
		};
		const utils::DMatrix<3, 3> H = {
			{ 6.0f, 26.0f, 46.0f	},
			{ 26.0f, 126.0f, 226.0f	},
			{ 46.0f, 226.0f, 406.0f	}
		};

		EXPECT_TRUE(E == G);
		EXPECT_TRUE(F == H);

		const utils::DMatrix<4, 3> I = {
			{ 1.0f, 1.0f, 1.0f, 1.0f	},
			{ 1.0f, 1.0f, 1.0f, 1.0f	},
			{ 1.0f, 1.0f, 1.0f, 1.0f	}
		};

		const utils::DMatrix<4, 3> J = {
			{ 1.0f, 2.0f, 3.0f, 4.0f	},
			{ 5.0f, 6.0f, 7.0f, 8.0f	},
			{ 9.0f, 10.0f, 11.0f, 12.0f	}
		};

		const utils::DMatrix<4, 1> K = {
			{ 1.0f, 2.0f, 3.0f, 4.0f	}
		};
		const utils::DMatrix<1, 3> L = {
			{ 1.0f	},
			{ 2.0f	},
			{ 3.0f	}
		};
		const utils::DMatrix<2, 3> M = {
			{ 1.0f, 4.0f	},
			{ 2.0f, 5.0f	},
			{ 3.0f, 6.0f	}
		};

		const utils::DMatrix<4, 3> R_IpJ = {
			{ 2.0f, 3.0f, 4.0f, 5.0f	},
			{ 6.0f, 7.0f, 8.0f, 9.0f, },
			{ 10.0f, 11.0f, 12.0f, 13.0f	}
		};
		const utils::DMatrix<4, 3> R_IpK = {
			{ 2.0f, 3.0f, 4.0f, 5.0f	},
			{ 2.0f, 3.0f, 4.0f, 5.0f	},
			{ 2.0f, 3.0f, 4.0f, 5.0f	}
		};
		const utils::DMatrix<4, 3> R_IpL = {
			{ 2.0f, 2.0f, 2.0f, 2.0f	},
			{ 3.0f, 3.0f, 3.0f, 3.0f	},
			{ 4.0f, 4.0f, 4.0f, 4.0f	}
		};

		EXPECT_TRUE(R_IpJ == I + J);
		EXPECT_TRUE(R_IpK == I + K);
		EXPECT_TRUE(R_IpL == I + L);
	}

	TEST(device_matrix_test, sum)
	{
		const utils::DMatrix<4, 3> A = {
			{ 1.0f, 2.0f, 3.0f, 4.0f	},
			{ 5.0f, 6.0f, 7.0f, 8.0f	},
			{ 9.0f, 10.0f, 11.0f, 12.0f	}
		};

		EXPECT_FLOAT_EQ(A.SumGet(), 78.0f);
		EXPECT_FLOAT_EQ(A.SumRowGet(0), 10.0f);
		EXPECT_FLOAT_EQ(A.SumRowGet(1), 26.0f);
		EXPECT_FLOAT_EQ(A.SumRowGet(2), 42.0f);
		EXPECT_FLOAT_EQ(A.SumColGet(0), 15.0f);
		EXPECT_FLOAT_EQ(A.SumColGet(1), 18.0f);
		EXPECT_FLOAT_EQ(A.SumColGet(2), 21.0f);
		EXPECT_FLOAT_EQ(A.SumColGet(3), 24.0f);
	}

	TEST(device_matrix_test, normalization)
	{
		const utils::DMatrix<4, 3> A = {
			{ 1.0f, 2.0f, 3.0f, 4.0f	},
			{ 5.0f, 6.0f, 7.0f, 8.0f	},
			{ 9.0f, 10.0f, 11.0f, 12.0f	}
		};

		const utils::DMatrix<4, 3> RFull = {
			{ 1.0f / 78.0f, 2.0f / 78.0f, 3.0f / 78.0f, 4.0f / 78.0f	},
			{ 5.0f / 78.0f, 6.0f / 78.0f, 7.0f / 78.0f, 8.0f / 78.0f	},
			{ 9.0f / 78.0f, 10.0f / 78.0f, 11.0f / 78.0f, 12.0f / 78.0f	}
		};
		const utils::DMatrix<4, 3> RRow = {
			{ 1.0f / 10.0f, 2.0f / 10.0f, 3.0f / 10.0f, 4.0f / 10.0f	},
			{ 5.0f / 26.0f, 6.0f / 26.0f, 7.0f / 26.0f, 8.0f / 26.0f	},
			{ 9.0f / 42.0f, 10.0f / 42.0f, 11.0f / 42.0f, 12.0f / 42.0f	}
		};
		const utils::DMatrix<4, 3> RCol = {
			{ 1.0f / 15.0f, 2.0f / 18.0f, 3.0f / 21.0f, 4.0f / 24.0f	},
			{ 5.0f / 15.0f, 6.0f / 18.0f, 7.0f / 21.0f, 8.0f / 24.0f	},
			{ 9.0f / 15.0f, 10.0f / 18.0f, 11.0f / 21.0f, 12.0f / 24.0f	}
		};

		utils::DMatrix<A.kColsN, A.kRowsN> B = A;
		B.NormalizeFull();

		EXPECT_TRUE(B == RFull);

		B = A;
		B.NormalizeByRow();
		EXPECT_TRUE(B == RRow);

		B = A;
		B.NormalizeByColumn();
		EXPECT_TRUE(B == RCol);
	}

} // Namespace test


/*
 *	--- End of file ---
 */
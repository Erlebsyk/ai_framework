/**
 * \file	main.cpp
 * \brief	
 * 
 * \author	Erlend Isachsen (erlend.isachsen@7sense.no)
 * 
*/

/** Standard library header includes	*/
#include <cstdint>

/** NRF / Zephyr header includes	*/


/** Project header includes	*/
#include <iostream>
#include <vector>
#include <cstdint>
#include <array>

#include "utils/matrix.cuh"

int main()
{
	std::cout << "Hello world!" << std::endl;

	math::DMatrix A(3,2);
	math::DMatrix B(2,3);

	uint32_t val = 1;
	for(uint32_t row_i = 0; row_i < A.n_rows_; row_i++)
	{
		for(uint32_t col_i = 0; col_i < A.n_cols_; col_i++)
		{
			A.Set(col_i, row_i, val);
			val++;
		}
	}
	for(uint32_t row_i = 0; row_i < B.n_rows_; row_i++)
	{
		for(uint32_t col_i = 0; col_i < B.n_cols_; col_i++)
		{
			B.Set(col_i, row_i, val);
			val++;
		}
	}
	std::cout << "Values set" << std::endl;
	math::Matrix h_A(A);
	math::Matrix h_B(B);
	std::cout << h_A << std::endl;
	std::cout << h_B << std::endl;

	math::DMatrix C(A * B);
	math::Matrix h_C(C);

	std::cout << h_C << std::endl;

	return 0;
}

/*
 *	--- End of file ---
 */
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

	utils::DMatrix<3,2> A;
	utils::DMatrix<2,3> B;

	uint32_t val = 1;
	for(uint32_t row_i = 0; row_i < A.kRowsN; row_i++)
	{
		for(uint32_t col_i = 0; col_i < A.kColsN; col_i++)
		{
			A.Set(col_i, row_i, val);
			val++;
		}
	}
	for(uint32_t row_i = 0; row_i < B.kRowsN; row_i++)
	{
		for(uint32_t col_i = 0; col_i < B.kColsN; col_i++)
		{
			B.Set(col_i, row_i, val);
			val++;
		}
	}
	std::cout << "Values set" << std::endl;

	return 0;
}

/*
 *	--- End of file ---
 */
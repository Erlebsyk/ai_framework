/*********************************************************************
 * \file   neural_net_test.cpp
 * \brief
 *
 * \author Erlend Isachsen
 *********************************************************************/

 /** Related header include */
#include "../src/utils/matrix.cuh"
#include "../src/ai/neural_net.cuh"

/** Standard library includes */
#include <cstdint>
#include <memory>
#include <gtest/gtest.h>

/** External library includes */

/** Project includes */


namespace test
{
	/** Declarations */


	/** Definitions */
	TEST(neural_net_test, construction)
	{
		CREATE_INPUT(input, 4, 3);

		ai::Layer<4, 3, 2> L(input, ai::ActivationFunction::kSoftMax);


	}

} // test

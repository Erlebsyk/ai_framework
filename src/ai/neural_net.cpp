/*********************************************************************
 * \file   neural_net.cpp
 * \brief  
 * 
 * \author Erlend Isachsen
 *********************************************************************/


/** Related header include */
#include "neural_net.hpp"

/** Standard library includes */
#include <cstdint>

/** External library includes */

/** Project includes */


namespace ai
{
	/** Declarations */


	/** Definitions */
	void Layer::AddBiases()
	{
		outputs_ += biases_;
	}

	float Layer::ActivationFunction_ReLU(const float x)
	{
		return (x > 0.0f) ? x : 0.0f;
	}

	float Layer::ActivationFunction_Sigmoid(const float x)
	{
		return 0.0f;
	}

	float Layer::ActivationFunction_Softmax(const float x)
	{
		for (size_t i = 0; i < outputs_.n_rows_; i++)
		{
			float row_sum;
			cublasSasum(nullptr, outputs_.n_cols_, outputs_.Get(), outputs_.n_rows_, &row_sum);

			//const float norm_factor = 1.0f / ();
		}
		
		return 0.0f;
	}

} // ai


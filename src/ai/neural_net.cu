/*********************************************************************
 * \file   neural_net.cpp
 * \brief  
 * 
 * \author Erlend Isachsen
 *********************************************************************/

#ifndef NEURAL_NET_CU_
#define NEURAL_NET_CU_

/** Related header include */
#include "neural_net.cuh"

/** Standard library includes */
#include <cstdint>

/** External library includes */

/** Project includes */

namespace ai
{
	/** Declarations */


	/** Definitions */
	template<size_t N_I, size_t N_S, size_t N_N>
	void Layer<N_I, N_S, N_N>::AddBiases()
	{
		outputs_ += biases_;
	}

	template<size_t N_I, size_t N_S, size_t N_N>
	float Layer<N_I, N_S, N_N>::ActivationFunction_ReLU(const float x)
	{
		return (x > 0.0f) ? x : 0.0f;
	}

	template<size_t N_I, size_t N_S, size_t N_N>
	float Layer<N_I, N_S, N_N>::ActivationFunction_Sigmoid(const float x)
	{
		return 0.0f;
	}

	template<size_t N_I, size_t N_S, size_t N_N>
	float Layer<N_I, N_S, N_N>::ActivationFunction_Softmax(const float x)
	{
		for (size_t i = 0; i < outputs_.n_rows_; i++)
		{
			float row_sum;
			cublasSasum(nullptr, outputs_.n_cols_, outputs_.Get(), outputs_.n_rows_, &row_sum);

			//const float norm_factor = 1.0f / ();
		}
		
		return 0.0f;
	}
	template<size_t N_I, size_t N_S, size_t N_N>
	Layer<N_I, N_S, N_N>::Layer(
		const std::shared_ptr<utils::DMatrix<N_I, N_S>> inputs,
		ActivationFunction activation_function
	) : 
		inputs_						{ inputs	},
		weights_					{			},
		biases_						{			},
		outputs_					{			},
		activation_function_ptr_	{ nullptr	}
	{

	}

	template<size_t N_I, size_t N_S, size_t N_N>
	Layer<N_I, N_S, N_N>::~Layer()
	{

	}
	template<size_t N_I, size_t N_S, size_t N_N>
	void Layer<N_I, N_S, N_N>::Propagate()
	{
		outputs_ = inputs_ * weights_;
		outputs_ = outputs_ + biases;
	}

} // ai

#endif // NEURAL_NET_CU_
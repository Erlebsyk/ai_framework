/*********************************************************************
 * \file   neural_net.hpp
 * \brief  
 * 
 * \author Erlend Isachsen
 *********************************************************************/

#ifndef NEURAL_NET_HPP_
#define NEURAL_NET_HPP_

/** Standard library includes */
#include <cstdint>
#include <memory>

/** External library includes */

/** Project includes */
#include "../utils/matrix.cuh"

namespace ai
{
	/** Declarations */
	#define CREATE_INPUT(name, inputs_n, sequence_n) \
		std::shared_ptr<utils::DMatrix<inputs_n, sequence_n>> name = std::make_shared<utils::DMatrix<inputs_n, sequence_n>>();

	template<size_t N_INPUTS, size_t N_NEURONS, size_t M> class Layer;
	class NeuralNet;

	/** Definitions */
	enum class ActivationFunction
	{
		kReLU,
		kSigmoid,
		kSoftMax
	};

	template<size_t N_INPUTS, size_t N_SERIES, size_t N_NEURONS>
	class Layer{
	protected:
		const std::shared_ptr<utils::DMatrix<N_INPUTS, N_SERIES>> inputs_;
		utils::DMatrix<N_NEURONS, N_INPUTS> weights_;
		utils::DMatrix<N_NEURONS, 1> biases_;
		utils::DMatrix<N_INPUTS, N_NEURONS> outputs_;
		void (*activation_function_ptr_)(float);

		void AddBiases();

		float ActivationFunction_ReLU(float x);
		float ActivationFunction_Sigmoid(float x);
		float ActivationFunction_Softmax(float x);

	public:
		Layer(
			const std::shared_ptr<utils::DMatrix<N_INPUTS, N_SERIES>> inputs,
			ActivationFunction activation_function
		);
		~Layer();

		void Propagate();

		utils::DMatrix<N_NEURONS, N_INPUTS> WeightsGet() const;
		utils::DMatrix<N_NEURONS, 1> BiasesGet() const;
		std::shared_ptr<const utils::DMatrix<N_INPUTS, N_NEURONS>> OutputGet() const;
	};
} // ai

#include "neural_net.cu"

#endif //NEURAL_NET_HPP_

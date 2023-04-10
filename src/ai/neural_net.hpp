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

/** External library includes */

/** Project includes */
#include "../utils/matrix.cuh"

namespace ai
{
	/** Declarations */
	class Layer;
	class NeuralNet;

	/** Definitions */
	enum class ActivationFunction
	{
		kReLU,
		kSigmoid,
		kSoftMax
	};

	class Layer{
	protected:
		const std::shared_ptr<utils::DMatrix> inputs_;
		utils::DMatrix weights_;
		utils::DMatrix biases_;
		utils::DMatrix outputs_;
		bool propagated_;
		void (*activation_function_ptr_)(float);

		void AddBiases();

		float ActivationFunction_ReLU(float x);
		float ActivationFunction_Sigmoid(float x);
		float ActivationFunction_Softmax(float x);

	public:
		Layer(
			const std::shared_ptr<utils::DMatrix> inputs,
			const size_t n_neurons,
			ActivationFunction activatiuon_function
		);
		~Layer();

		void Propagate();

		utils::DMatrix WeightsGet() const;
		utils::DMatrix BiasesGet() const;
		std::shared_ptr<const utils::DMatrix> OutputGet() const;
	};

} // ai

#endif //NEURAL_NET_HPP_

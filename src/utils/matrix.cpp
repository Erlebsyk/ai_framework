/*********************************************************************
 * \file   matrix.cpp
 * \brief  
 * 
 * \author Erlend Isachsen
 *********************************************************************/

/** Related header include */
#include "matrix.cuh"

/** Standard library includes */
#include <string>
#include <sstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/** External library includes */

/** Project includes */


namespace utils
{
	/** Declarations */


	/** Definitions */
	void cuda_error_parser(
		std::string caller_f,
		std::string called_f,
		cudaError_t error,
		std::string context
	)
	{
		if (cudaSuccess != error)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << caller_f << "]";

			if (context != "")
				err_msg << " with context [" << context << "]. ";
			else
				err_msg << ". ";

			err_msg << "The function [" << called_f << "] returned with error code [" << error << "].";
			throw(std::runtime_error(err_msg.str()));
		}
	}

	void cublas_error_parser(
		std::string caller_f,
		std::string called_f,
		cublasStatus_t status,
		std::string context
	)
	{
		if (CUBLAS_STATUS_SUCCESS != status)
		{
			std::stringstream err_msg;
			err_msg << "Runtime error in [" << caller_f << "]";

			if (context != "")
				err_msg << " with context [" << context << "]. ";
			else
				err_msg << ". ";

			err_msg << "The function [" << called_f << "] returned with status code [" << status << "].";
			throw(std::runtime_error(err_msg.str()));
		}
	}

} // utils
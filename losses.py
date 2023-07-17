def log1pMSELoss(log_predicted_counts, true_counts):
	"""A MSE loss on the log(x+1) of the inputs.

	This loss will accept tensors of predicted counts and a vector of true
	counts and return the MSE on the log of the labels. The squared error
	is calculated for each position in the tensor and then averaged, regardless
	of the shape.

	Note: The predicted counts are in log space but the true counts are in the
	original count space.

	Parameters
	----------
	log_predicted_counts: torch.tensor, shape=(n, ...)
		A tensor of log predicted counts where the first axis is the number of
		examples. Important: these values are already in log space.

	true_counts: torch.tensor, shape=(n, ...)
		A tensor of the true counts where the first axis is the number of
		examples.

	Returns
	-------
	loss: torch.tensor, shape=(n, 1)
		The MSE loss on the log of the two inputs, averaged over all examples
		and all other dimensions.
	"""

	log_true = torch.log(true_counts+1)
	return torch.mean(torch.square(log_true - log_predicted_counts), dim=-1)
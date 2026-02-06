/*
 * GreyML C API header: ga loss.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once
#include "ga_tensor.h"

GA_API GATensor* ga_mse_loss(GATensor* pred, GATensor* target, int reduction);
GA_API GATensor* ga_l1_loss(GATensor* pred, GATensor* target, int reduction);
GA_API GATensor* ga_cross_entropy_loss(GATensor* logits, GATensor* target, int reduction);
GA_API GATensor* ga_nll_loss(GATensor* log_probs, GATensor* target, int reduction);
GA_API GATensor* ga_binary_cross_entropy(GATensor* pred, GATensor* target, int reduction);
GA_API GATensor* ga_huber_loss(GATensor* pred, GATensor* target, float delta, int reduction);

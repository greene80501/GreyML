/*
 * GreyML backend: ga loss ce.
 *
 * Built-in loss implementations intended for training loops and optimizer tests.
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include "greyarea/ga_loss.h"
#include "greyarea/ga_ops.h"
#include "greyarea/ga_common.h"

// Cross-entropy implemented in ga_loss_other via ga_cross_entropy_loss; keep file to satisfy build

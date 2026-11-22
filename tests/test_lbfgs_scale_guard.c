#include "lbfgs.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int expect_rejected(const real_t* s, const real_t* y, real_t norm_residual)
{
    unsigned int previous_size = lbfgs_get_active_buffer_size();
    if (lbfgs_update(s, y, norm_residual) != FAILURE ||
        lbfgs_get_active_buffer_size() != previous_size) {
        fprintf(stderr, "invalid secant pair changed the L-BFGS memory\n");
        return 1;
    }
    return 0;
}

static int check_dimension(unsigned int n, int accept_cancellation)
{
    real_t* s = calloc(n, sizeof(real_t));
    real_t* y = calloc(n, sizeof(real_t));
    const real_t* direction;
    unsigned int i;
    int status;

    if (s == NULL || y == NULL) return 1;
    if (lbfgs_init(5, n) == FAILURE) return 2;
    /* Same nonzero components, padded with zeros. The accumulation bound
     * must reflect the actual dot-product length, not a fixed coefficient. */
    s[0] = s[1] = 1.0;
    y[0] = 1.0; y[1] = -1.0 + 128.0 * DBL_EPSILON;
    status = lbfgs_update(s, y, 0.0);
    if ((status == SUCCESS) != accept_cancellation) {
        fprintf(stderr, "unexpected cancellation decision at dimension %u\n", n);
        return 3;
    }
    lbfgs_reset();
    /* Well-resolved tiny curvature must still be accepted in high dimension. */
    for (i = 0; i < n; ++i) { s[i] = 1e-11; y[i] = 2e-11; }
    if (lbfgs_update(s, y, 1e-8) == FAILURE) return 4;
    direction = lbfgs_get_direction(s);
    for (i = 0; i < n; ++i) {
        if (!isfinite(direction[i]) || fabs(direction[i] + 0.5 * s[i]) > 1e-23) return 5;
    }
    lbfgs_cleanup();
    free(s); free(y);
    return 0;
}

int main(void)
{
    real_t reference[2], s[2], y[2];
    real_t residual[2] = {0.3, -0.4};
    const real_t scales[] = {1.0, 1e-6, 1e-11, 1e-100, 1e100};
    const real_t* direction;
    unsigned int i, j;

    if (lbfgs_init(5, 2) == FAILURE) return 1;
    for (i = 0; i < sizeof(scales) / sizeof(scales[0]); ++i) {
        lbfgs_reset();
        s[0] = scales[i]; s[1] = 2.0 * scales[i];
        y[0] = 2.0 * scales[i]; y[1] = 8.0 * scales[i];
        if (lbfgs_update(s, y, 1e-8) == FAILURE) {
            fprintf(stderr, "valid pair rejected at scale %.3g\n", scales[i]);
            return 2;
        }
        direction = lbfgs_get_direction(residual);
        for (j = 0; j < 2; ++j) {
            if (!isfinite(direction[j])) return 3;
            if (i == 0) reference[j] = direction[j];
            else if (fabs(direction[j] - reference[j]) > 1e-12) return 4;
        }
    }

    /* Rejections must preserve the previously accepted pair. */
    s[0] = 1e-11; s[1] = 0.0; y[0] = -1e-11; y[1] = 0.0;
    if (expect_rejected(s, y, 1e-8)) return 5;
    y[0] = 0.0; y[1] = 1e-11;
    if (expect_rejected(s, y, 1e-8)) return 6;
    y[0] = NAN;
    if (expect_rejected(s, y, 1e-8)) return 7;
    y[0] = INFINITY;
    if (expect_rejected(s, y, 1e-8)) return 8;
    s[0] = 0.0; y[0] = 1.0;
    if (expect_rejected(s, y, 1e-8)) return 9;

    /* Positive but numerically unresolved curvature. */
    s[0] = 1.0; s[1] = 1.0; y[0] = 1.0; y[1] = -1.0 + DBL_EPSILON;
    if (expect_rejected(s, y, 1e-8)) return 10;
    /* Keep the existing residual-dependent curvature floor. */
    s[0] = 1.0; s[1] = 0.0; y[0] = 1e-6; y[1] = 1e-6;
    if (expect_rejected(s, y, 1e8)) return 11;
    /* Finite vectors whose curvature reciprocal or squared norm overflows. */
    s[0] = 1e-155; y[0] = 1e-155; y[1] = 0.0;
    if (expect_rejected(s, y, 1e-8)) return 12;
    s[0] = 1e200; y[0] = 1e200;
    if (expect_rejected(s, y, 1e-8)) return 13;
    s[0] = 1.0; y[0] = 2.0;
    if (expect_rejected(s, y, NAN) || expect_rejected(s, y, INFINITY) ||
        expect_rejected(s, y, -1.0)) return 15;

    direction = lbfgs_get_direction(residual);
    for (j = 0; j < 2; ++j) {
        if (!isfinite(direction[j]) || fabs(direction[j] - reference[j]) > 1e-12) return 14;
    }
    lbfgs_cleanup();
    if (check_dimension(2, 1) || check_dimension(20, 1) ||
        check_dimension(128, 0) || check_dimension(1024, 0)) return 16;
    puts("L-BFGS scale guard passed");
    return 0;
}

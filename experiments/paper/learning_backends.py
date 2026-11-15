from .original_safepdp_backends import OriginalForwardCasadiKKT, OriginalForwardSafePDP


def build_learning_backend(method, model, instance, tolerance):
    if method == 'casadi':
        return OriginalForwardCasadiKKT(model, instance, 1e-5, tolerance, cached_symbolic=True)
    if method == 'safepdp':
        return OriginalForwardSafePDP(model, instance, 1e-5, tolerance,
                                      reverse_variant='bounds_symmetric', cached_symbolic=True)
    raise ValueError(method)

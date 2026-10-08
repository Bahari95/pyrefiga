# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: M. BAHARI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#-------------------------------------------------------------------------------------------------------
# ... Nitsche's method for assembling the matrices : from paper https://hal.science/hal-01338133/document
# ===============================================================
# different spaces for FE and mapping can be implemented here
# ===============================================================
# diagonal matrix with respect to given patch
#--------------------------------------------------------------------------------------------------------
def assemble_interface_quadrature(
    jumps: 'float[:,:]', fluxes: 'float[:,:]', weights: 'float[:]',
    Kappa: 'float', normalS: 'float', matrices: 'float[:,:,:]'
):
    """Symmetric Nitsche form using physical normals and signed traces.

    jumps = (trace_source, -trace_neighbor); fluxes uses the source
    outward normal on both sides. Edge numbering is handled before this
    kernel, so the same form applies to all sixteen edge pairs.
    """
    for q in range(jumps.shape[0]):
        for i in range(jumps.shape[1]):
            for j in range(jumps.shape[1]):
                matrices[q, i, j] = weights[q] * (
                    Kappa * jumps[q, i] * jumps[q, j]
                    - normalS * (jumps[q, i] * fluxes[q, j]
                                 + fluxes[q, i] * jumps[q, j])
                )
from pyrefiga import bsplines
from pyrefiga import cad
from pyrefiga import spaces
from pyrefiga import linalg
from pyrefiga import quadratures
from pyrefiga import utilities
from pyrefiga import api
from pyrefiga import results
from pyrefiga import results_f90
from pyrefiga import ad_mesh_tools
from pyrefiga import fast_diag
from pyrefiga import nurbs_utilities

__all__ = ['bsplines', 'cad',
           'spaces', 'linalg',
           'quadratures', 'utilities', 'results', 'ad_mesh_tools', 'results_f90', 'api', 'nurbs_utilities']

from pyrefiga.bsplines import ( find_span,
                                 basis_funs,
                                 basis_funs_1st_der,
                                 basis_funs_all_ders,
                                 collocation_matrix,
                                 histopolation_matrix,
                                 breakpoints,
                                 greville,
                                 elements_spans,
                                 make_knots,
                                 elevate_knots,
                                 quadrature_grid,
                                 basis_integrals,
                                 basis_ders_on_quad_grid,
                                 scaling_matrix,
                                 hrefinement_matrix )

from pyrefiga.cad import ( point_on_bspline_curve,
                            point_on_nurbs_curve,
                            insert_knot_bspline_curve,
                            insert_knot_nurbs_curve,
                            elevate_degree_bspline_curve,
                            elevate_degree_nurbs_curve,
                            point_on_bspline_surface,
                            point_on_nurbs_surface,
                            insert_knot_bspline_surface,
                            insert_knot_nurbs_surface,
                            elevate_degree_bspline_surface,
                            elevate_degree_nurbs_surface,
                            translate_bspline_curve,
                            translate_nurbs_curve,
                            rotate_bspline_curve,
                            rotate_nurbs_curve,
                            homothetic_bspline_curve,
                            homothetic_nurbs_curve,
                            translate_bspline_surface,
                            translate_nurbs_surface,
                            homothetic_nurbs_curve,
                            translate_bspline_surface,
                            translate_nurbs_surface,
                            rotate_bspline_surface,
                            rotate_nurbs_surface,
                            homothetic_bspline_surface,
                            homothetic_nurbs_surface)

from pyrefiga.spaces import ( SplineSpace,
                               TensorSpace )

from pyrefiga.linalg import ( StencilVectorSpace,
                               StencilVector,
                               StencilMatrix )

from pyrefiga.quadratures import gauss_legendre

from pyrefiga.utilities import ( plot_field_1d,
                                  plot_field_2d,
                                  prolongation_matrix,
                                  build_dirichlet,
                                  getGeometryMap,
                                  save_geometry_to_xml)

from pyrefiga.results import ( sol_field_2d)

from pyrefiga.ad_mesh_tools import ( quadratures_in_admesh,
                                     assemble_stiffness1D,
                                     assemble_mass1D,
                                     assemble_matrix_ex01,
                                     assemble_matrix_ex02)

from pyrefiga.fast_diag import ( Poisson)

from pyrefiga.results_f90 import ( pyccel_sol_field_2d,
                                   pyccel_sol_field_1d,
                                    pyccel_sol_field_3d, 
                                    least_square_Bspline,
                                    plot_SolutionMultipatch,
                                    plot_MeshMultipatch,
                                    plot_AdMeshMultipatch,
                                    plot_FunctMultipatch,
                                    plot_JacobianMultipatch,
                                    paraview_AdMeshMultipatch,
                                    paraview_SolutionMultipatch)

from pyrefiga.api import (assemble_matrix, assemble_vector, assemble_scalar, compile_kernel, apply_dirichlet, apply_periodic, apply_zeros)

from pyrefiga.nurbs_utilities import(sol_field_NURBS_2d, sol_field_NURBS_3d, 
                                      prolongate_NURBS_mapping, least_square_NURBspline,
                                      paraview_nurbsAdMeshMultipatch, paraview_nurbsSolutionMultipatch,
                                      ViewGeo)
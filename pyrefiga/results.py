import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
import numpy                        as     np
from   numpy                        import empty
colors      = ['b', 'k', 'r', 'g', 'm', 'c', 'y', 'orange']
markers     = ['v', 'o', 's', 'D', '^', '<', '>', '*']  # Different markers
line_styles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (3, 2, 1, 2))]  # Different line styles 

# ==========================================================
def find_span( knots, degree, x ):
    # Knot index at left/right boundary
    low  = degree
    high = 0
    high = len(knots)-1-degree

    # Check if point is exactly on left/right boundary, or outside domain
    if x <= knots[low ]: returnVal = low
    elif x >= knots[high]: returnVal = high-1
    else:
        # Perform binary search
        span = (low+high)//2
        while x < knots[span] or x >= knots[span+1]:
            if x < knots[span]:
                high = span
            else:
                low  = span
            span = (low+high)//2
        returnVal = span

    return returnVal
# ==========================================================
def all_bsplines( knots, degree, x, span ):
    left   = empty( degree  , dtype=float )
    right  = empty( degree  , dtype=float )
    values = empty( degree+1, dtype=float )

    values[0] = 1.0
    for j in range(0,degree):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            temp      = values[r] / (right[r] + left[j-r])
            values[r] = saved + right[r] * temp
            saved     = left[j-r] * temp
        values[j+1] = saved

    return values

def basis_funs_all_ders( knots, degree, x, span, n ):
    """
    Evaluate value and n derivatives at x of all basis functions with
    support in interval [x_{span-1}, x_{span}].

    ders[i,j] = (d/dx)^i B_k(x) with k=(span-degree+j),
                for 0 <= i <= n and 0 <= j <= degree+1.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : float
        Evaluation point.

    span : int
        Knot span index.

    n : int
        Max derivative of interest.

    Results
    -------
    ders : numpy.ndarray (n+1,degree+1)
        2D array of n+1 (from 0-th to n-th) derivatives at x of all (degree+1)
        non-vanishing basis functions in given span.

    Notes
    -----
    The original Algorithm A2.3 in The NURBS Book [1] is here improved:
        - 'left' and 'right' arrays are 1 element shorter;
        - inverse of knot differences are saved to avoid unnecessary divisions;
        - innermost loops are replaced with vector operations on slices.

    """
    left  = np.empty( degree )
    right = np.empty( degree )
    ndu   = np.empty( (degree+1, degree+1) )
    a     = np.empty( (       2, degree+1) )
    ders  = np.zeros( (     n+1, degree+1) ) # output array

    # Number of derivatives that need to be effectively computed
    # Derivatives higher than degree are = 0.
    ne = min( n, degree )

    # Compute nonzero basis functions and knot differences for splines
    # up to degree, which are needed to compute derivatives.
    # Store values in 2D temporary array 'ndu' (square matrix).
    ndu[0,0] = 1.0
    for j in range(0,degree):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            # compute inverse of knot differences and save them into lower triangular part of ndu
            ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
            # compute basis functions and save them into upper triangular part of ndu
            temp       = ndu[r,j] * ndu[j+1,r]
            ndu[r,j+1] = saved + right[r] * temp
            saved      = left[j-r] * temp
        ndu[j+1,j+1] = saved

    # Compute derivatives in 2D output array 'ders'
    ders[0,:] = ndu[:,degree]
    for r in range(0,degree+1):
        s1 = 0
        s2 = 1
        a[0,0] = 1.0
        for k in range(1,ne+1):
            d  = 0.0
            rk = r-k
            pk = degree-k
            if r >= k:
               a[s2,0] = a[s1,0] * ndu[pk+1,rk]
               d = a[s2,0] * ndu[rk,pk]
            j1 = 1   if (rk  > -1 ) else -rk
            j2 = k-1 if (r-1 <= pk) else degree-r
            a[s2,j1:j2+1] = (a[s1,j1:j2+1] - a[s1,j1-1:j2]) * ndu[pk+1,rk+j1:rk+j2+1]
            d += np.dot( a[s2,j1:j2+1], ndu[rk+j1:rk+j2+1,pk] )
            if r <= pk:
               a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
               d += a[s2,k] * ndu[r,pk]
            ders[k,r] = d
            j  = s1
            s1 = s2
            s2 = j

    # Multiply derivatives by correct factors
    r = degree
    for k in range(1,ne+1):
        ders[k,:] = ders[k,:] * r
        r = r * (degree-k)

    return ders

# ==========================================================
def point_on_bspline_surface(Tu, Tv, P, u, v):
    pu = len(Tu) - P.shape[0] - 1
    pv = len(Tv) - P.shape[1] - 1
    d = P.shape[-1]

    span_u = find_span( Tu, pu, u )
    span_v = find_span( Tv, pv, v )

    basis_x =basis_funs_all_ders( Tu, pu, u, span_u, 1 )
    basis_y =basis_funs_all_ders( Tv, pv, v, span_v, 1 )

    bu   = basis_x[0,:]
    bv   = basis_y[0,:]
    
    derbu   = basis_x[1,:]
    derbv   = basis_y[1,:]
        
    c = np.zeros(d)
    cx = np.zeros(d)
    cy = np.zeros(d)
    for ku in range(0, pu+1):
        for kv in range(0, pv+1):
            c[:] += bu[ku]*bv[kv]*P[span_u-pu+ku, span_v-pv+kv,:]
            cx[:] += derbu[ku]*bv[kv]*P[span_u-pu+ku, span_v-pv+kv,:]
            cy[:] += bu[ku]*derbv[kv]*P[span_u-pu+ku, span_v-pv+kv,:]
    return c, cx, cy

from numpy import zeros, linspace, meshgrid, asarray
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sol_field_2d(Npoints,  uh , knots, degree):
    # Using computed control points U we compute solution
    # in new discretisation by Npoints    
    pu, pv = degree
    Tu, Tv = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1    

    if Npoints is None:

      nx = nu-pu+1
      ny = nv-pv+1
    
      xs = Tu[pu:-pu] #linspace(Tu[pu], Tu[-pu-1], nx)
    
      ys = Tv[pv:-pv] #linspace(Tv[pv], Tv[-pv-1], ny)
      
    else :
      nx, ny = Npoints

      xs = linspace(Tu[pu], Tu[-pu-1], nx)
    
      ys = linspace(Tv[pv], Tv[-pv-1], ny)

    P = zeros((nu, nv,1))     
    
    i = list(range(nu))
    for j in range(nv):
        P[i, j, 0] = uh[i,j]    

    Q  = zeros((nx, ny, 3))
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            Q[i,j,:]   = point_on_bspline_surface(Tu, Tv, P, x, y)
    X, Y = meshgrid(xs, ys)

    return Q[:,:,0], Q[:,:,1], Q[:,:,2], X, Y

# ==========================================================
def point_on_bspline_curve(knots, P, x):
    degree = len(knots) - len(P) - 1
    d = P.shape[-1]

    span = find_span( knots, degree, x )
    b    = basis_funs_all_ders(knots, degree, x, span, 0)

    c = np.zeros(d)
    for k in range(0, degree+1):
        c[:] += b[k,0]*P[span-degree+k,:]
    return c

# ==========================================================
def plot_field_1d(knots, degree, u, nx=101, color='b', xmin = None, xmax = None, label = None, plot = False):
    n = len(knots) - degree - 1

    if xmin is None :
        xmin = knots[degree]
    if xmax is None :
        xmax = knots[-degree-1]

    xs = np.linspace(xmin, xmax, nx)

    P = np.zeros((len(u), 1))
    P[:,0] = u[:]
    Q = np.zeros((nx, 1))
    for i,x in enumerate(xs):
        Q[i,:] = point_on_bspline_curve(knots, P, x)

    if label is not None :
        plt.plot(xs, Q[:,0], label = label)
    else :
        plt.plot(xs, Q[:,0])
    if plot:
        plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_results(X, Y, xlabel = '$\\mathbf{Time}$', ylabel = '$\\mathbf{H^1-error}$', MyLabel = '$\\mathbf{Error}$', mylocname = 'figs/error', 
             xscale = True, yscale = True, lw = 2.5, i =0, j = 3, legend = True,
             font_size = 13, markersize = 10, axes_size = 12, grid = True, margins = (0.02,0.02), plot = False):
   '''
   plot error under refinement or over time 
   '''
   font = {'family': 'serif', 
            'color':  'k', 
            'weight': 'normal', 
            'size': font_size, 
            }    
   fig, axes =plt.subplots() 
   for i in range(len(X)):
      print('---',i)
      plt.plot( X[i], Y[i], color=colors[i], lw = lw, ls=line_styles[j], marker=markers[i], markersize = markersize, markerfacecolor = colors[i], label = MyLabel[i])
   if xscale:
      plt.xscale('log')
   if yscale:    
     plt.yscale('log')
   if xlabel is not None:
      plt.xlabel(xlabel,  fontweight ='bold', fontdict=font)
   if ylabel is not None:
      plt.ylabel(ylabel,  fontweight ='bold', fontdict=font)
   if grid:
      plt.grid(color='k', linestyle='--', linewidth=0.5, which ="both")
   plt.margins(margins[0], margins[1])
   if legend:
      plt.legend(fontsize=font_size)
   plt.tick_params(axis='both', which='major', labelsize=axes_size)
   fig.tight_layout()
   # Set axes (ticks) font weight
   for label in axes.get_xticklabels() + axes.get_yticklabels():
      label.set_fontweight('bold') 
   plt.savefig(mylocname+'.png')
   plt.show(block=plot)


'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_SolutionMultipatch(nbpts, xuh, V, V_geo, xmp, ymp, savefig = None, plot = True): 
   """
   Plot the solution of the problem in the whole multi-patch domain
   """
   #---Compute a solution
   numPaches = len(xmp)
   u   = []
   F1  = []
   F2  = []
   for i in range(numPaches):
      u.append(pyccel_sol_field_2d((nbpts, nbpts), xuh[i], V.knots, V.degree)[0])
      #---Compute a solution
      F1.append(pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V_geo.knots, V_geo.degree)[0])
      F2.append(pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V_geo.knots, V_geo.degree)[0])

   # --- Compute Global Color Levels ---
   u_min  = min(np.min(u[0]), np.min(u[1]))
   u_max  = max(np.max(u[0]), np.max(u[1]))
   for i in range(2, numPaches):
      u_min  = min(u_min, np.min(u[i]))
      u_max  = max(u_max, np.max(u[i]))
   levels = np.linspace(u_min, u_max+1e-10, 100)  # Uniform levels for both plots

   # --- Create Figure ---
   fig, axes = plt.subplots(figsize=(8, 6))

   # --- Contour Plot for First Subdomain ---
   im = []
   for i in range(numPaches):
      im.append(axes.contourf(F1[i], F2[i], u[i], levels, cmap='jet'))
      # --- Colorbar ---
      divider = make_axes_locatable(axes)
      cax = divider.append_axes("right", size="5%", pad=0.05, aspect=40)
      cbar = plt.colorbar(im[i], cax=cax)
      cbar.ax.tick_params(labelsize=15)
      cbar.ax.yaxis.label.set_fontweight('bold')
   # --- Formatting ---
   axes.set_title("Numerical Solution", fontweight='bold')
   for label in axes.get_xticklabels() + axes.get_yticklabels():
      label.set_fontweight('bold')

   fig.tight_layout()
   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_JacobianMultipatch(nbpts, V, xmp, ymp, savefig = None, plot = True): 
   """
   Plot the solution of the problem in the whole domain
   """
   #---Compute a solution
   numPaches = len(V)
   u   = []
   F1  = []
   F2  = []
   for i in range(numPaches):
      #---Compute a solution
      F1.append(pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0])
      F2.append(pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0])
      #...Compute a Jacobian
      F1x, F1y = pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[1:3]
      F2x, F2y = pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[1:3]
      u.append(F1x*F2y - F1y*F2x)

   # --- Compute Global Color Levels ---
   u_min  = min(np.min(u[0]), np.min(u[1]))
   u_max  = max(np.max(u[0]), np.max(u[1]))
   for i in range(2, numPaches):
      u_min  = min(u_min, np.min(u[i]))
      u_max  = max(u_max, np.max(u[i]))
   levels = np.linspace(u_min, u_max+1e-10, 100)  # Uniform levels for both plots

   # --- Create Figure ---
   fig, axes = plt.subplots(figsize=(8, 6))

   # --- Contour Plot for First Subdomain ---
   im = []
   for i in range(numPaches):
      im.append(axes.contourf(F1[i], F2[i], u[i], levels, cmap='jet'))
      # --- Colorbar ---
      divider = make_axes_locatable(axes)
      cax = divider.append_axes("right", size="5%", pad=0.05, aspect=40)
      cbar = plt.colorbar(im[i], cax=cax)
      cbar.ax.tick_params(labelsize=15)
      cbar.ax.yaxis.label.set_fontweight('bold')
   # --- Formatting ---
   #axes.set_title("Jacobian the in whole domain ", fontweight='bold')
   for label in axes.get_xticklabels() + axes.get_yticklabels():
      label.set_fontweight('bold')

   fig.tight_layout()
   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_MeshMultipatch(nbpts, V, xmp, ymp, cp = True, savefig = None, plot = True): 
   """
   Plot the solution of the problem in the whole domain
   """
   #---Compute a solution
   numPaches = len(V)
   F1 = []
   F2 = []
   for i in range(numPaches):
      #---Compute a mesh
      F1.append(pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0])
      F2.append(pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0])

   # --- Create Figure ---
   fig =plt.figure() 
   # ---
   for ii in range(numPaches):
      #---------------------------------------------------------
      for i in range(nbpts):
         phidx = F1[ii][:,i]
         phidy = F2[ii][:,i]

         plt.plot(phidx, phidy, linewidth = 0.5, color = 'k')
      for i in range(nbpts):
         phidx = F1[ii][i,:]
         phidy = F2[ii][i,:]

         plt.plot(phidx, phidy, linewidth = 0.5, color = 'k')
      if cp:
         plt.plot(xmp[ii].reshape(V[ii].nbasis[0]*V[ii].nbasis[1]), ymp[ii].reshape(V[ii].nbasis[0]*V[ii].nbasis[1]), 'ro', markersize=3.5)
      #~~~~~~~~~~~~~~~~~~~~
      #.. Plot the surface
      if ii == 1:
         phidx = F1[ii][:,0]
         phidy = F2[ii][:,0]
         plt.plot(phidx, phidy, '--k', linewidth=2., label = '$Im([0,1]^2_{y=0})$')
         # ...
         phidx = F1[ii][:,nbpts-1]
         phidy = F2[ii][:,nbpts-1]
         plt.plot(phidx, phidy, '-g', linewidth=2. ,label = '$Im([0,1]^2_{y=1})$')
      else :
         phidx = F1[ii][:,0]
         phidy = F2[ii][:,0]
         plt.plot(phidx, phidy, '-g', linewidth=2., label = '$Im([0,1]^2_{y=0})$')
         # ...
         phidx = F1[ii][:,nbpts-1]
         phidy = F2[ii][:,nbpts-1]
         plt.plot(phidx, phidy, '--k', linewidth=2. ,label = '$Im([0,1]^2_{y=1})$')
      #''
      phidx = F1[ii][0,:]
      phidy = F2[ii][0,:]
      plt.plot(phidx, phidy, '-r',  linewidth=2., label = '$Im([0,1]^2_{x=0})$')
      # ...
      phidx = F1[ii][nbpts-1,:]
      phidy = F2[ii][nbpts-1,:]
      plt.plot(phidx, phidy, '-r', linewidth= 2., label = '$Im([0,1]^2_{x=1}$)')

   #axes[0].axis('off')
   plt.margins(0,0)

   fig.tight_layout()
   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_FunctMultipatch(nbpts, V, xmp, ymp, functions, cp = True, savefig = None, plot = True): 
   """
   Plot the function in the whole domain
   """
   #---Compute a solution
   numPaches = len(V)
   F1     = []
   F2     = []
   values = []
   for i in range(numPaches):
      #---Compute a mesh
      F1.append(pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0])
      F2.append(pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0])
      values.append(functions(F1[i], F2[i]))

   # --- Compute Global Color Levels ---
   u_min  = min(np.min(values[0]), np.min(values[1]))
   u_max  = max(np.max(values[0]), np.max(values[1]))
   for i in range(2, numPaches):
      u_min  = min(u_min, np.min(values[i]))
      u_max  = max(u_max, np.max(values[i]))
   levels = np.linspace(u_min, u_max+1e-10, 100)  # Uniform levels for both plots
   # --- Create Figure ---
   # ... Analytic Density function
   fig, axes =plt.subplots() 
   for i in range(numPaches):
      im2 = plt.contourf( F1[i], F2[i], values[i], levels, cmap= 'plasma')
   #divider = make_axes_locatable(axes) 
   #cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
   #plt.colorbar(im2, cax=cax) 
   fig.tight_layout()

   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_AdMeshMultipatch(nbpts, V, xmp, ymp, xad, yad, cp = True, savefig = None, plot = True, patchesInterface = False): 
   """
   Plot the solution of the problem in the whole domain
   """
   #---Compute a solution
   numPaches = len(V)
   F1 = []
   F2 = []
   for i in range(numPaches):
      sx = pyccel_sol_field_2d((nbpts, nbpts), xad[i], V[i].knots, V[i].degree)[0]
      sy = pyccel_sol_field_2d((nbpts, nbpts), yad[i], V[i].knots, V[i].degree)[0]
      #---Compute a mesh
      F1.append(pyccel_sol_field_2d((None, None), xmp[i], V[i].knots, V[i].degree, mesh=(sx, sy))[0])
      F2.append(pyccel_sol_field_2d((None, None), ymp[i], V[i].knots, V[i].degree, mesh=(sx, sy))[0])

   # --- Create Figure ---
   fig =plt.figure() 

   # ---
   for ii in range(numPaches):
      #---------------------------------------------------------
      for i in range(nbpts):
         phidx = F1[ii][:,i]
         phidy = F2[ii][:,i]

         plt.plot(phidx, phidy, linewidth = 0.3, color = 'k')
      for i in range(nbpts):
         phidx = F1[ii][i,:]
         phidy = F2[ii][i,:]

         plt.plot(phidx, phidy, linewidth = 0.3, color = 'k')
      if cp:
         plt.plot(xmp[ii].reshape(V[ii].nbasis[0]*V[ii].nbasis[1]), ymp[ii].reshape(V[ii].nbasis[0]*V[ii].nbasis[1]), 'ro', markersize=3.5)
      #~~~~~~~~~~~~~~~~~~~~
      #.. Plot the surface
      if patchesInterface:
         phidx = F1[ii][:,0]
         phidy = F2[ii][:,0]
         plt.plot(phidx, phidy, '--k', linewidth=0.25, label = '$Im([0,1]^2_{y=0})$')
         # ...
         phidx = F1[ii][:,nbpts-1]
         phidy = F2[ii][:,nbpts-1]
         plt.plot(phidx, phidy, '--k', linewidth=0.25 ,label = '$Im([0,1]^2_{y=1})$')

         phidx = F1[ii][:,0]
         phidy = F2[ii][:,0]
         plt.plot(phidx, phidy, '--k', linewidth=0.25, label = '$Im([0,1]^2_{y=0})$')
         # ...
         phidx = F1[ii][:,nbpts-1]
         phidy = F2[ii][:,nbpts-1]
         plt.plot(phidx, phidy, '--k', linewidth=0.25,label = '$Im([0,1]^2_{y=1})$')
         #''
         phidx = F1[ii][0,:]
         phidy = F2[ii][0,:]
         plt.plot(phidx, phidy, '--k',  linewidth=0.25, label = '$Im([0,1]^2_{x=0})$')
         # ...
         phidx = F1[ii][nbpts-1,:]
         phidy = F2[ii][nbpts-1,:]
         plt.plot(phidx, phidy, '--k', linewidth= 0.25, label = '$Im([0,1]^2_{x=1}$)')

   #axes[0].axis('off')
   plt.margins(0,0)

   fig.tight_layout()
   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0

   '''
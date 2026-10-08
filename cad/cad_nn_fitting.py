"""
cad_nn_fitting.py

Fit CAD boundary curves using a small feed-forward neural network.

For now only C0 fitting is supported: the network learns f(t) = (x(t), y(t))
from sampled boundary points (function-value / C0 matching only, no explicit
derivative/tangent-continuity constraints). The trained model exposes a
`predict` method and can be passed directly (as a callable) to
`pyrefiga.least_square_Bspline` to project the fitted curve onto a B-spline
space, so it plugs into the same workflow as `cadboundary.py`.

@author : M. BAHARI
"""

import argparse

import matplotlib.pyplot as plt
import numpy             as np

from pyrefiga import least_square_Bspline
from pyrefiga import SplineSpace


#==============================================================================
class MLP:
    """A minimal fully-connected feed-forward neural network (numpy only).

    Trained with full-batch gradient descent + Adam, `tanh` hidden
    activations and a linear output layer (suitable for curve regression).
    """

    def __init__(self, layer_sizes, seed=0):
        rng          = np.random.default_rng(seed)
        self.sizes   = list(layer_sizes)
        self.weights = []
        self.biases  = []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            scale = np.sqrt(2.0/n_in)
            self.weights.append(rng.normal(0., scale, size=(n_in, n_out)))
            self.biases.append(np.zeros(n_out))

    #...
    def forward(self, X):
        """Returns (activations, pre_activations) for all layers."""
        a  = X
        As = [a]
        Zs = []
        n_layers = len(self.weights)
        for l, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a.dot(W) + b
            Zs.append(z)
            # ... linear output layer, tanh for hidden layers
            a = z if l == n_layers-1 else np.tanh(z)
            As.append(a)
        return As, Zs

    #...
    def predict(self, t):
        t = np.asarray(t, dtype=float).reshape(-1, 1)
        As, _ = self.forward(t)
        return As[-1]

    #...
    def train(self, X, Y, epochs=3000, lr=1e-2, verbose=False):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        m = X.shape[0]

        # ... Adam optimizer state
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mW = [np.zeros_like(W) for W in self.weights]
        vW = [np.zeros_like(W) for W in self.weights]
        mb = [np.zeros_like(b) for b in self.biases]
        vb = [np.zeros_like(b) for b in self.biases]

        n_layers = len(self.weights)
        for epoch in range(1, epochs+1):
            As, Zs = self.forward(X)
            y_pred = As[-1]

            # ... MSE loss gradient
            dA = 2.0*(y_pred - Y)/m

            gW = [None]*n_layers
            gb = [None]*n_layers
            for l in reversed(range(n_layers)):
                dZ    = dA if l == n_layers-1 else dA*(1.0 - As[l+1]**2)
                gW[l] = As[l].T.dot(dZ)
                gb[l] = dZ.sum(axis=0)
                dA    = dZ.dot(self.weights[l].T)

            for l in range(n_layers):
                mW[l] = beta1*mW[l] + (1-beta1)*gW[l]
                vW[l] = beta2*vW[l] + (1-beta2)*gW[l]**2
                mb[l] = beta1*mb[l] + (1-beta1)*gb[l]
                vb[l] = beta2*vb[l] + (1-beta2)*gb[l]**2

                mW_hat = mW[l]/(1-beta1**epoch)
                vW_hat = vW[l]/(1-beta2**epoch)
                mb_hat = mb[l]/(1-beta1**epoch)
                vb_hat = vb[l]/(1-beta2**epoch)

                self.weights[l] -= lr*mW_hat/(np.sqrt(vW_hat)+eps)
                self.biases[l]  -= lr*mb_hat/(np.sqrt(vb_hat)+eps)

            if verbose and (epoch % max(1, epochs//10) == 0):
                loss = np.mean((y_pred - Y)**2)
                print(f'epoch {epoch:6d}/{epochs}  mse = {loss:.3e}')


#==============================================================================
def fit_curve_c0(t, values, hidden_layers=(32, 32), epochs=3000, lr=1e-2,
                  seed=0, verbose=False):
    """Fits `values = f(t)` with an MLP (C0 / function-value fitting only).

    Parameters
    ----------
    t          : 1D array of parameter samples in [0, 1]
    values     : array of shape (len(t),) or (len(t), d) of target values
    Returns
    -------
    model : MLP instance with a `predict(t)` method
    """
    t      = np.asarray(t, dtype=float).reshape(-1, 1)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    model = MLP([1, *hidden_layers, values.shape[1]], seed=seed)
    model.train(t, values, epochs=epochs, lr=lr, verbose=verbose)
    return model


#==============================================================================
def sample_boundary(expr_x, expr_y, n=200):
    """Samples a parametric curve x(t), y(t), t in [0, 1] given as strings."""
    t = np.linspace(0., 1., n)
    x = eval(expr_x, {'t': t, 'np': np, 'pi': np.pi})
    y = eval(expr_y, {'t': t, 'np': np, 'pi': np.pi})
    return t, np.asarray(x, dtype=float), np.asarray(y, dtype=float)


#==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='C0 fitting of a CAD boundary curve using a neural network.')
    parser.add_argument('--expr_x', type=str, default='np.cos(2.*pi*t)',
                         help="x(t) expression, e.g. 'np.cos(2.*pi*t)'")
    parser.add_argument('--expr_y', type=str, default='np.sin(2.*pi*t)',
                         help="y(t) expression, e.g. 'np.sin(2.*pi*t)'")
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--hidden', type=int, nargs='+', default=[32, 32])
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--degree', type=int, default=3, help='B-spline degree used for the projection')
    parser.add_argument('--nelements', type=int, default=16, help='number of elements used for the projection')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    # ... sample the analytic boundary and train the network
    t, x, y = sample_boundary(args.expr_x, args.expr_y, n=args.n_samples)
    model   = fit_curve_c0(t, np.stack([x, y], axis=1),
                            hidden_layers=tuple(args.hidden),
                            epochs=args.epochs, lr=args.lr, verbose=True)

    # ... project the NN-fitted curve onto a B-spline space (control points)
    V     = SplineSpace(degree=args.degree, nelements=args.nelements)
    xc    = least_square_Bspline(V.degree, V.knots, lambda s: float(model.predict(s)[0, 0]))
    yc    = least_square_Bspline(V.degree, V.knots, lambda s: float(model.predict(s)[0, 1]))

    if args.plot:
        t_fine = np.linspace(0., 1., 500)
        xy_nn  = model.predict(t_fine)
        plt.plot(x, y, 'k.', label='samples')
        plt.plot(xy_nn[:, 0], xy_nn[:, 1], 'r-', label='NN fit (C0)')
        plt.plot(xc, yc, 'bo--', label='B-spline control points')
        plt.legend()
        plt.axis('equal')
        plt.show()

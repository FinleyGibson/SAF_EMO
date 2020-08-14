import numpy as np
import GPy
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import sys
current_module = sys.modules[__name__]


class MonoSurrogate:
    def __init__(self, x, y, scaled=False):
        """
        Parent class for surrogates used in optimisation

        :param x[np.array]: real-valued input data to real function, shape (n_datum, data_dimensions)
        :param y[np.array]: real-valued output data from real function, shape (n_datum, n_objectives)
        """
        if x.ndim == 2:
            self.x = x
        else:
            self.x = x.reshape(1,-1)
        if y.ndim == 2:
            self.y = y
        else:
            self.y = y.reshape(1,-1)

        assert isinstance(scaled, bool)
        self.scaled = scaled
        self.x_dims = x.shape[1]
        self.n_obj = y.shape[1]
        self.model = None

        if self.scaled:
            self.mean_x = np.mean(x, axis=0)
            self.mean_y = np.mean(y, axis=0)
            self.std_x = np.std(x, axis=0)
            self.std_y = np.std(y, axis=0)

        try:
            self.update(x, y)
        except AttributeError:
            print("Inputs values to GP should be np.array() of shape (n_points, dimensions)")

    def update(self, x, y):
        """
        updates the surrogate, taking real-valued data and making a surrogate with the data scaled to mean=0, variance=1.

        :param x[np.array]: real-valued input data to real function, shape (n_datum, data_dimensions)
        :param y[np.array]: real-valued output data from real function, shape (n_datum, n_objectives)
        """
        raise NotImplementedError

    def predict(self, xi):
        """
        makes surrogate prediction of the output for F at location xi

        :param xi[np.ndarray]: real-vaued query point in parameter space.
        :return[tuple(np.ndarray, np.ndarray)]: real-valued mean prediction and variance
        """
        raise NotImplementedError

    def scale_x(self, x):
        """scales x to a range between -1 and 1, with a mean of 0"""
        if self.scaled:
            return (x - self.mean_x)/self.std_x
        else:
            return x

    def scale_y(self, y):
        """scales x to a range between -1 and 1, with a mean of 0"""
        if self.scaled:
            return (y-self.mean_y)/self.std_y
        else:
            return y

    def descale_x(self, x):
        """undoes self.scale_x"""
        if self.scaled:
            return (x*self.std_x)+self.mean_x
        else:
            return x

    def descale_y(self, y):
        """undoes self.scale_y"""
        if self.scaled:
            return (y*self.std_y)+self.mean_y
        else:
            return y

    def get_x(self):
        """passes out the real-valued x by passing it through self.descale_x"""
        return self.descale_x(self.x)

    def get_y(self):
        """passes out the real-valued y by passing it through self.descale_y"""
        return self.descale_y(self.y)


class GP(MonoSurrogate):
    def __init__(self, x, y, kernel=None, scaled=True):
        """
        Class for Gaussian Process surrogates.

        :param x[np.array]: real-valued input data to real function, shape (n_datum, data_dimensions)
        :param y[np.array]: real-valued output data from real function, shape (n_datum, n_objectives)
        :param kernel: Gpy kernel object, leave unspecified to use Matern52
        :param scaled[bool]: if True data is scaled to std=1, mean=0 before fitting GP. This generally
        improves performance.
        """
        self.kernel = kernel
        super().__init__(x=x, y=y, scaled=scaled)

    def update(self, x, y):
        """
        updates the GP, taking real-valued data and making a GP with the data scaled to mean=0, variance=1.

        :param x[np.array]: real-valued input data to real function, shape (n_datum, data_dimensions)
        :param y[np.array]: real-valued output data from real function, shape (n_datum, n_objectives)
        """
        # check dimensions of data
        assert(x.ndim == 2)
        assert(y.ndim == 2)
        assert(x.shape[0] == y.shape[0])

        # scale data
        self.mean_x = np.mean(x, axis=0)
        self.std_x = np.std(x, axis=0)
        self.x = self.scale_x(x)
        self.x_dims = x.shape[1]

        self.mean_y = np.mean(y, axis=0)
        self.std_y = np.std(y, axis=0)
        self.y = self.scale_y(y)
        self.n_obj = y.shape[1]

        if self.kernel is None:
            self.kernel = GPy.kern.Matern52(input_dim=self.x_dims, variance=1, lengthscale=0.2, ARD=False)
        else:
            pass

        self.model = GPy.models.GPRegression(self.x, self.y, self.kernel)
        self.model['.*lengthscale'].constrain_bounded(1e-5, 1e5)
        self.model['.*variance'].constrain_bounded(1e-5, 1e5)
        self.model['.*noise'].constrain_fixed(1e-20)
        self._fit_kernel()

    def predict(self, xi):
        """
        makes GP prediction of the output for F at location xi

        :param xi[np.ndarray]: real-vaued query point in parameter space.
        :return[tuple(np.ndarray, np.ndarray)]: real-valued mean prediction and variance
        """
        xi = self.scale_x(xi)

        try:
            mean, var = self.model.predict(xi)
        except AssertionError:
            mean, var = self.model.predict(xi.reshape(1, -1))

        return self.descale_y(mean), var*self.std_y**2

    def _fit_kernel(self):
        """
        re-fits kernel forrGP
        """
        try:
            self.model.optimize_restarts(messages=False, num_restarts=10)
        except AttributeError:
            print("Must supply values for x and y before kernel can be fit. try {}.update(x,y)".format(self))


class RF(MonoSurrogate):
    def __init__(self, x, y, extra_trees=True, n_trees=100):
        """
        Class for random forest surrogates.

        :param x[np.array]: real-valued input data to real function, shape (n_datum, data_dimensions)
        :param y[np.array]: real-valued output data from real function, shape (n_datum, n_objectives)
        """
        self.extra_trees = extra_trees
        self.n_trees = n_trees
        super().__init__(x=x, y=y, scaled=False)

    def update(self, x, y):
        """
        updates the GP, taking real-valued data and making a GP with the data scaled to mean=0, variance=1.

        :param x[np.array]: real-valued input data to real function, shape (n_datum, data_dimensions)
        :param y[np.array]: real-valued output data from real function, shape (n_datum, n_objectives)
        """
        # check dimensions of data
        assert(x.ndim == 2)
        assert(y.ndim == 2)
        assert(x.shape[0] == y.shape[0])

        self.x = self.scale_x(x)
        self.x_dims = x.shape[1]

        self.y = self.scale_y(y)
        self.n_obj = y.shape[1]

        if self.extra_trees:
            self.model = ExtraTreesRegressor(n_estimators=self.n_trees)
        else:
            self.model = RandomForestRegressor(n_estimators=self.n_trees)

        # fit model, accounting for sklearn preference for 1d y data where possible
        if y.ndim == 2 and y.shape[1] == 1:
            self.model.fit(self.x, self.y.flatten())
        else:
            self.model.fit(self.x, self.y)

    def predict(self, xi):
        """
        makes GP prediction of the output for F at location xi

        :param xi[np.ndarray]: real-vaued query point in parameter space.
        :return[tuple(np.ndarray, np.ndarray)]: real-valued mean prediction and variance
        """
        xi = self.scale_x(xi)

        try:
            mean = self.model.predict(xi)
        except ValueError:
            mean, = self.model.predict(xi.reshape(-1,1))

        # return mean predictions with zero variance
        mean = np.array(mean)
        return mean, np.zeros_like(mean)


class Surrogate:
    """
    creates an instance of MonoSurrogate for each of the required surrogates to model the objective. One instance of
    MonoSurrogate for a mono-surrogate approach to the MOP, and n instances for a multi-surrogate approach to an n
    objective problem
    """
    def __init__(self, x, y, surrogate_type, *args, multi_surrogate=False, **kwargs):
        self.multi_surrogate = multi_surrogate
        self.surrogate_class = getattr(current_module, surrogate_type)
        self.x_dims = x.shape[1]
        self.n_obj = y.shape[1]

        # establish surrogate/surrogates as instances of self.surrogate_class
        if self.multi_surrogate:
            self.surrogate = [self.surrogate_class(x, y[:, i:i+1], *args, **kwargs) for i in range(y.shape[1])]
        else:
            self.surrogate = [self.surrogate_class(x, y, *args, **kwargs)]

    def update(self, x, y):
        if self.multi_surrogate:
            for i, surr in enumerate(self.surrogate):
                surr.update(x, y[:, i:i+1])
        else:
            self.surrogate.update(x, y)

    def predict(self, xi):
        """calls predict methods from surrogates and returns them as a numpy.array
        
        :param xi[np.ndarray]: real-valued query point to be predicted by surrogates.

        :returns [np.ndarray]: array of mean predictions and variances for each objective in the mop.
        shape [2, n_queries, n_objectives]
        """
        if self.multi_surrogate:
            predictions = np.asarray([surr.predict(xi) for surr in self.surrogate])
            # to handle extra axis in GP return.
            try:
                return np.moveaxis(predictions.squeeze(-1), 0, -1)
            except ValueError:
                return np.moveaxis(predictions, 0, -1)

        else:
            return np.asarray(self.surrogate.predict(xi))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    def test_function(x):
        return np.array([[np.sum([np.sin(xii) for xii in xi]) for xi in x],
                         [np.sum([np.cos(xii) for xii in xi]) for xi in x],
                         [np.sum([np.cos(xii)**2 for xii in xi]) for xi in x]]).T

    def plot_surrogate_vs_true(surrogate, title):
        pred_mean, pred_var = surrogate.predict(x)
        fig0 = plt.figure()
        ax0 = fig0.gca()
        ax0.plot(x, y[:,0], c="C0", alpha=0.4)
        ax0.plot(x, y[:,1], c="C1", alpha=0.4)
        ax0.plot(x, y[:,2], c="C2", alpha=0.4)
        ax0.plot(x, pred_mean[:,0], c="C0", linestyle="--", label="objective 1")
        ax0.plot(x, pred_mean[:,1], c="C1", linestyle="--", label="objective 2")
        ax0.plot(x, pred_mean[:,2], c="C2", linestyle="--", label="objective 3")
        ax0.scatter(xtr, ytr[:,0], c="C0", marker="x")
        ax0.scatter(xtr, ytr[:,1], c="C1", marker="x")
        ax0.scatter(xtr, ytr[:,2], c="C2", marker="x")
        plt.legend()
        ax0.set_title(title)


    x = np.linspace(50, 55, 50).reshape(-1, 1)
    y = test_function(x)

    xtr = np.random.uniform(50, 55, size=(10, 1))
    ytr = test_function(xtr)

    gp_mono_scaled = GP(xtr, ytr, scaled=True)
    plot_surrogate_vs_true(gp_mono_scaled, title="g mono surrogate")

    rf_mono_extra = RF(xtr, ytr, n_trees=200, extra_trees=True)
    plot_surrogate_vs_true(rf_mono_extra, title="rf mono surrogate")

    gp_surr_multi = Surrogate(xtr, ytr, surrogate_type="GP", multi_surrogate=True)
    plot_surrogate_vs_true(gp_surr_multi, title="gp multi surrogate")

    rf_surr_multi = Surrogate(xtr, ytr, surrogate_type="RF", multi_surrogate=True, n_trees=100, extra_trees=True)
    plot_surrogate_vs_true(rf_surr_multi, title="rf multi surrogate")

    plt.show()

    pass

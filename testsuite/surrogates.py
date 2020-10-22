import numpy as np
import GPy
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import sys
current_module = sys.modules[__name__]

class MonoSurrogate:
    def __init__(self, scaled=False):
        """
        Parent class for surrogates used in optimisation

        :param x[np.array]: real-valued input data to real function, shape (n_datum, data_dimensions)
        :param y[np.array]: real-valued output data from real function, shape (n_datum, n_objectives)
        """
        assert isinstance(scaled, bool)
        self.scaled = scaled
        self.x = None
        self.y = None
        self.x_dims = None
        self.n_objectives = None
        self.model = None

        if scaled:
            self.mean_x = None
            self.std_x = None
            self.mean_y = None
            self.std_y = None

    def update(self, x, y):
        """
        :param np.array x: decision vector (n_datum, input_dimensions)
        :param np.array y: objective function evaluations OF(x)
        (n_datum, n_objectives)
        """
        if x.ndim<2:
            x = y.reshape(-1, 1)
        if y.ndim<2:
            y = y.reshape(-1, 1)
        # check dimensions of data
        assert(x.ndim == 2)
        assert(y.ndim == 2)
        assert(x.shape[0] == y.shape[0])

        if self.scaled:
            # scale data
            self.mean_x = np.mean(x, axis=0)
            self.std_x = np.std(x, axis=0)
            self.x = self.scale_x(x)

            self.mean_y = np.mean(y, axis=0)
            self.std_y = np.std(y, axis=0)
            self.y = self.scale_y(y)
        else:
            self.x = x
            self.y = y

        self.x_dims = self.x.shape[1]
        self.n_objectives = self.y.shape[1]

    def predict(self, xi):
        """
        makes surrogate prediction of the output for F at location xi

        :param np.ndarray xi: unscaled query point in parameter space.
        :return (np.ndarray, np.ndarray): mean prediction and variance
        """
        if xi.ndim == 0:
            # handles instance where xi is 0 dimensional array
            xi = xi.reshape(1,1)
        elif xi.ndim == 1:
            # handles instance where xi is 1 dimensional array
            xi = xi.reshape(1, -1)

        xi = self.scale_x(xi)
        n_queries = xi.shape[0]

        try:
            mean, var = self.model.predict(xi)
        except ValueError:
            # handle predictions which do not provide variance information
            # reshape ensures mean is 2 dimensional in all cases
            mean = self.model.predict(xi).reshape(n_queries, -1)
            var = np.zeros_like(mean)
        except AttributeError as err:
            print("Must choose a defined surrogate, not blank {} "
                  "instance".format(__class__.__name__))
            raise err

        if var.shape[1] != mean.shape[1]:
            # handles instances of single variance for all predictions
            var = np.tile(var, mean.shape[1]).reshape(xi.shape[0],
                                                      mean.shape[1])

        if self.scaled:
            return self.descale_y(mean), var*self.std_y**2
        else:
            return mean, var

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
    def __init__(self, kernel=None, scaled=True):
        """
        Class for Gaussian Process surrogates.

        :param kernel: Gpy kernel object, leave unspecified for Matern52
        :param scaled[bool]: if True data is scaled to std=1, mean=0
        before fitting GP. This generally improves performance.
        """
        self.kernel = kernel
        super().__init__(scaled=scaled)

    def update(self, x, y):
        """
        updates the GP, taking real-valued data and generating a GP and
        optimising the kernel parameters according to maximum liklihood.

        :param np.array x: decision vectors (n_vectors, vector_dims)
        :param np.array y: objective function evealuations of x
        (n_vectors, n_objectives)
        """
        super().update(x, y)    # basic update from MonoSurrogate

        if self.kernel is None:
            # default Matern52 kernel
            self.kernel = GPy.kern.Matern52(input_dim=self.x_dims, variance=1,
                                            lengthscale=0.2, ARD=False)
        else:
            pass

        self.model = GPy.models.GPRegression(self.x, self.y, self.kernel)
        self.model['.*lengthscale'].constrain_bounded(1e-5, 1e5)
        self.model['.*variance'].constrain_bounded(1e-5, 1e5)
        self.model['.*noise'].constrain_fixed(1e-20)
        self._fit_kernel()

    def _fit_kernel(self):
        """
        re-fits kernel for GP
        """
        try:
            self.model.optimize_restarts(messages=False, num_restarts=10)
        except AttributeError:
            print("Must supply values for x and y before kernel can be"
                  "fit. try {}.update(x,y)".format(self))


class RF(MonoSurrogate):
    def __init__(self, extra_trees=True, n_trees=100):
        """
        class for random forest surrogates.

        :param bool extra_trees:
        :param int n_trees:
        """
        self.extra_trees = extra_trees
        self.n_trees = n_trees
        super().__init__(scaled=False)

    def update(self, x, y):
        """
        updates the RF, taking real-valued data and generating a GP and
        optimising the kernel parameters according to maximum liklihood.

        :param np.array x: decision vectors (n_vectors, vector_dims)
        :param np.array y: objective function evealuations of x
        (n_vectors, n_objectives)
        """
        super().update(x, y)

        if self.extra_trees:
            self.model = ExtraTreesRegressor(n_estimators=self.n_trees)
        else:
            self.model = RandomForestRegressor(n_estimators=self.n_trees)

        # fit model
        if y.ndim == 2 and y.shape[1] == 1:
            self.model.fit(self.x, self.y.flatten())
        else:
            self.model.fit(self.x, self.y)


class MultiSurrogate:
    def __init__(self, surrogate, *args, **kwargs):
        self.surrogate_model = surrogate
        self.mono_surrogates = None
        self.x = None
        self.y = None
        self.x_dims = None
        self.n_objectives = None
        self.surrogate_args = args
        self.surrogate_kwargs = kwargs

    def update(self, x, y):

        # check dimensions of data
        assert(x.ndim == 2)
        assert(y.ndim == 2)
        assert(x.shape[0] == y.shape[0])

        if not self.mono_surrogates:
            # instantiate surrogates if they do not already exist
            self.mono_surrogates = [
                self.surrogate_model(*self.surrogate_args,
                                     **self.surrogate_kwargs)
                for i in range(y.shape[1])]

        for i, surrogate in enumerate(self.mono_surrogates):
            surrogate.update(x, y[:, i:i+1])

        self.x = x
        self.y = y

        self.x_dims = self.x.shape[1]
        self.n_objectives = self.y.shape[1]

    def predict(self, xi):
        predictions = np.array([surrogate.predict(xi) for surrogate in
                                self.mono_surrogates]).squeeze(-1).swapaxes(1, 2).T

        return predictions[0], predictions[1]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def test_function(x):
        return np.array([[np.sum([np.sin(xii) for xii in xi]) for xi in x],
                         [np.sum([np.cos(xii) for xii in xi]) for xi in x],
                         [np.sum([np.cos(xii)**2 for xii in xi]) for xi in x]
                         ]).T

    def plot_surrogate_vs_true(surrogate, title):
        pred_mean, pred_var = surrogate.predict(x)
        fig0 = plt.figure()
        ax0 = fig0.gca()
        ax0.plot(x, y[:, 0], c="C0", alpha=0.4)
        ax0.plot(x, y[:, 1], c="C1", alpha=0.4)
        ax0.plot(x, y[:, 2], c="C2", alpha=0.4)
        ax0.plot(x, pred_mean[:, 0], c="C0", linestyle="--", label="obj 1")
        ax0.plot(x, pred_mean[:, 1], c="C1", linestyle="--", label="obj 2")
        ax0.plot(x, pred_mean[:, 2], c="C2", linestyle="--", label="obj 3")
        ax0.scatter(xtr, ytr[:, 0], c="C0", marker="x")
        ax0.scatter(xtr, ytr[:, 1], c="C1", marker="x")
        ax0.scatter(xtr, ytr[:, 2], c="C2", marker="x")
        plt.legend()
        ax0.set_title(title)


    x = np.linspace(50, 55, 50).reshape(-1, 1)
    y = test_function(x)

    xtr = np.random.uniform(50, 55, size=(10, 1))
    ytr = test_function(xtr)

    gp_mono_scaled = GP(scaled=True)
    gp_mono_scaled.update(xtr, ytr)
    answer_gp_mono = gp_mono_scaled.predict(x)
    plot_surrogate_vs_true(gp_mono_scaled, title="gp mono surrogate")

    rf_mono_extra = RF(n_trees=200, extra_trees=True)
    rf_mono_extra.update(xtr, ytr)
    answer_rf_mono = rf_mono_extra.predict(x)[0]
    plot_surrogate_vs_true(rf_mono_extra, title="rf mono surrogate")

    gp_surr_multi = MultiSurrogate(GP, scaled=True)
    gp_surr_multi.update(xtr, ytr)
    answer_gp_multi = gp_surr_multi.predict(x)[0]
    plot_surrogate_vs_true(gp_surr_multi, title="gp multi surrogate")

    rf_surr_multi = MultiSurrogate(RF, n_trees=200, extra_trees=True)
    rf_surr_multi.update(xtr, ytr)
    answer_rf_multi = rf_surr_multi.predict(x)[0]
    plot_surrogate_vs_true(rf_surr_multi, title="rf multi surrogate")
    print("--"*20)
    real = test_function(x)
    print("real: ", real)
    print("--"*20)
    print(answer_gp_mono)
    print("--"*20)
    print(answer_rf_mono)
    print("--"*20)
    print(answer_gp_multi)
    print("--"*20)
    print(answer_rf_multi)

    plt.show()
    pass
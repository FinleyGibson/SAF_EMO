import numpy as np
import GPy
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


class Surrogate:
    def __init__(self, x=None, y=None, scaled=False):
        """
        Parent class for surrogates used in optimisation

        :param x[np.array]: real-valued input data to real function, shape (n_datum, data_dimensions)
        :param y[np.array]: real-valued output data from real function, shape (n_datum, n_objectives)
        """
        self.scaled = scaled
        assert isinstance(self.scaled, bool)
        if x is None:
            self.x = None
            self.y = None
            self.n_dims = None
            self.n_obj = None
            self.model = None

            if self.scaled:
                self.mean_x = None
                self.mean_y = None
                self.std_x = None
                self.std_y = None
        else:
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


class GP(Surrogate):
    def __init__(self, x=None, y=None, kernel=None, scaled=True):
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
        self.n_dims = x.shape[1]

        self.mean_y = np.mean(y, axis=0)
        self.std_y = np.std(y, axis=0)
        self.y = self.scale_y(y)
        self.n_obj = y.shape[1]

        if self.kernel is None:
            self.kernel = GPy.kern.Matern52(input_dim=self.n_dims, variance=1, lengthscale=0.2, ARD=False)
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


class RF(Surrogate):
    def __init__(self, x=None, y=None, extra_trees=True, n_trees=100):
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
        self.n_dims = x.shape[1]

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
            mean, = self.model.predict(xi.reshape(1,-1))

        # return mean predictions with zero variance
        mean = np.array(mean).reshape((-1,1))
        return mean, np.zeros_like(mean)


if __name__ == "__main__":
    x = np.random.randn(10, 5)*10+100
    y = np.random.randn(x.shape[0]).reshape(-1,1)


    # non scaled version
    c = GP(x, y, scaled=False)

    # test scaling
    a = GP(x, y)
    np.testing.assert_almost_equal(np.std(a.x, axis=0), np.ones_like(x[0]))
    np.testing.assert_almost_equal(np.std(a.y, axis=0), np.ones_like(y[0]))
    np.testing.assert_almost_equal(np.mean(a.x, axis=0), np.zeros_like(x[0]))
    np.testing.assert_almost_equal(np.mean(a.y, axis=0), np.zeros_like(y[0]))
    np.testing.assert_array_almost_equal(x, a.get_x())
    np.testing.assert_array_almost_equal(y, a.get_y())
    np.testing.assert_array_almost_equal(a.x, a.scale_x(x))
    np.testing.assert_array_almost_equal(a.y, a.scale_y(y))
    np.testing.assert_array_almost_equal(x, c.get_x())
    np.testing.assert_array_almost_equal(y, c.get_y())
    b = GP()
    b.update(x, y)

    d = RF(x, y)
    np.testing.assert_array_almost_equal(d.get_x(), x)
    np.testing.assert_array_almost_equal(d.get_y(), y)
    np.testing.assert_array_almost_equal(d.x, x)
    np.testing.assert_array_almost_equal(d.y, y)

    # case testing
    # 1 prediction
    test_x = np.random.randn(5)
    a_ans = a.predict(test_x)
    b_ans = b.predict(test_x)
    c_ans = c.predict(test_x)
    d_ans = d.predict(test_x)

    np.testing.assert_array_almost_equal(a_ans, b_ans)
    np.testing.assert_array_equal(np.shape(a_ans), np.shape(b_ans))
    np.testing.assert_array_equal(np.shape(a_ans), np.shape(c_ans))
    np.testing.assert_array_equal(np.shape(a_ans), np.shape(d_ans))


    # n prediction
    test_x = np.random.randn(2,5)
    a_ans = a.predict(test_x)
    b_ans = b.predict(test_x)
    c_ans = c.predict(test_x)
    d_ans = d.predict(test_x)

    np.testing.assert_array_almost_equal(a_ans, b_ans)
    np.testing.assert_array_equal(np.shape(a_ans), np.shape(b_ans))
    np.testing.assert_array_equal(np.shape(a_ans), np.shape(c_ans))
    np.testing.assert_array_equal(np.shape(a_ans), np.shape(d_ans))



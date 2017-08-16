from pymc3.distributions import transforms
import theano.tensor as T


class Reciprocal(transforms.ElemwiseTransform):

    name = "Reciprocal"

    def backward(self, x):
        return T.exp(x)

    def forward(self, x):
        return 1. / x

    def forward_val(self, x, point=None):
        return self.forward(x)

    def jacobian_det(self, x):
        return x


class Exp(transforms.ElemwiseTransform):

    name = "Exponential"

    def backward(self, x):
        return T.log(x)

    def forward(self, x):
        return T.exp(x)

    def forward_val(self, x, point=None):
        return self.forward(x)

    def jacobian_det(self, x):
        return x
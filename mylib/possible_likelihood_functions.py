import random
import math

class SigmoidPolynomial:
    def __init__(self, degree, coeff_min=0.1, coeff_max=1.0):
        """
        Create a polynomial of given degree with random positive coefficients.
        Output will be passed through a sigmoid function.
        """
        self.degree = degree
        self.coeffs = [random.uniform(coeff_min, coeff_max) for _ in range(degree+1)]
        self.coeffs[0]=0

    def __str__(self):
        """Return polynomial in human-readable form."""
        terms = []
        for i, coeff in enumerate(self.coeffs):
            if i == 0:
                terms.append(f"{coeff:.2f}")
            elif i == 1:
                terms.append(f"{coeff:.2f}*x")
            else:
                terms.append(f"{coeff:.2f}*x^{i}")
        return " + ".join(terms)

    def update_coeffs(self, new_coeffs):
        """Update coefficients (must be positive)."""
        if len(new_coeffs) != len(self.coeffs[1:]):
            raise ValueError("Length of new_coeffs must match polynomial degree + 1")
        if any(c < 0 for c in new_coeffs):
            raise ValueError("All coefficients must be positive")
        self.coeffs[1:] = new_coeffs

    def _poly(self, x):
        """Compute raw polynomial value at x."""
        return sum(c * (x**i) for i, c in enumerate(self.coeffs))

    def evaluate(self, x):
        """Return sigmoid(polynomial(x))."""
        z = self._poly(x)
        return 2 / (1 + math.exp(z))   # range from 0 to 1 when z is positive
    

#ob=SigmoidPolynomial(degree=2, coeff_min=0.1, coeff_max=1.0)
#print(ob.degree)
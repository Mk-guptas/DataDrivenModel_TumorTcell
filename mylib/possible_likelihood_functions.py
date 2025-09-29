import random
import math
import numpy as np

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



class InteractionPolynomialSigmoid:
    def __init__(self, degree=2, seed=None):
        self.degree = degree
        self.rng = np.random.default_rng(seed)
        
        # Build list of exponents (i, j) where i,j >=1
        # These vanish if x1=0 or x2=0
        self.exponents = [(i, j) for i in range(1, degree+1)
                                   for j in range(1, degree+1)]
        
        
        # Initialize coefficients randomly positive
        self.coeffs = np.abs(self.rng.normal(loc=1.0, scale=0.5, size=len(self.exponents)))
    
    def update_coeffs(self, new_coeffs):
        """
        Update coefficients manually.
        new_coeffs : array-like
            Must match number of terms.
        """
        new_coeffs = np.array(new_coeffs, dtype=float)
        if new_coeffs.shape[0] != len(self.coeffs[:]):
            raise ValueError(f"Expected {len(self.coeffs)} coefficients, got {len(new_coeffs)}")
        self.coeffs[:] = new_coeffs

    def polynomial(self, x1, x2):
        """
        Evaluate polynomial part z(x1, x2).
        """
        z = 0.0
        for (coef, (i, j)) in zip(self.coeffs, self.exponents):
            z += coef * (x1**i) * (x2**j)
        return z
    
    def evaluate(self, x1, x2):
        """
        Evaluate sigmoid(s(z(x1, x2))).
        """
        z = self.polynomial(x1, x2)
        return 2.0 / (1.0 + np.exp(z))
    
    def num_terms(self):
        return len(self.exponents)
    
    def describe(self):
        """
        Print polynomial structure.
        """
        terms = []
        for coef, (i, j) in zip(self.coeffs, self.exponents):
            if (i, j) == (0,0):
                terms.append(f"{coef:.3f}")
            else:
                terms.append(f"{coef:.3f} * x1^{i} * x2^{j}")
        return " + ".join(terms)




#ob1=InteractionPolynomialSigmoid(degree=2, seed=42)
#ob1.update_coeffs([0.5, 0.5, 0.5, 0.5])
##print(ob1.describe())

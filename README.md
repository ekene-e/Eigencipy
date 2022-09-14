## Eigencipy
### Overview and Documentation
Please ignore the cheesy name; it's derived from an inside joke (which I'll tell you about if you want to know :)).

Anyhow, this is a mini-library written in C++ for all your linear algebra needs, or at least all 
non-trivially implementable ones I am aware of, including but not limited to
* slicing and splicing sections of the matrix,
* [the dot product](https://en.wikipedia.org/wiki/Dot_product)
* [the Rayleigh quotient](https://en.wikipedia.org/wiki/Rayleigh_quotient)
* [the Frobenius norm](https://mathworld.wolfram.com/FrobeniusNorm.html)
* [Hermitian transposition](https://en.wikipedia.org/wiki/Conjugate_transpose)
* [QU decomposition](https://en.wikipedia.org/wiki/QR_decomposition)
* [Back-substitution and Forward-substitution](https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution)
* [Givens rotations](https://en.wikipedia.org/wiki/Givens_rotation)
* [Upper Hessenberg](https://en.wikipedia.org/wiki/Hessenberg_matrix#Upper_Hessenberg_matrix), et cetera.

Documentation for the use of utility $X$ is included in C++ docstrings right above $X$.

### References
First of all, the giant that is Wikipedia. I am discovering with increasing regularity the sheer value of the maths on Wikipedia, and I
am immensely grateful that such a resource is out there.

Second, a great book by Horn and Garcia, _A Second Course in Linear Algebra_, which I found very helpful for truly learning  to a further
 extent a lot of the extensions of the linear algebra implemented in this library.
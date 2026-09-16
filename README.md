# Sylvester

An exact linear algebra workbench for the terminal.

Sylvester covers a full first course in linear algebra and shows its work at
every step: the row operation in textbook notation, the theorem the conclusion
rests on, and a line stating what was verified. Nothing is rounded. `1/3` stays
`1/3`, and an eigenvalue of `(1 + √5)/2` stays `(1 + √5)/2` — including when its
eigenvector is computed.

No dependencies. Python standard library only.

Named for James Joseph Sylvester, who coined both *matrix* and *nullity*.

```
$ python -m sylvester

 SYLVESTER   home
 an exact linear algebra workbench

  Registers
  A 3x3   B 3x1   U 3x1   V 3x1

  ▸ 1  Matrix registers               edit A-Z, load samples
    2  Row reduction                  CLO 2  REF, RREF, every operation
    3  Linear systems                 CLO 1  solve, classify, general solution
    4  Determinants                   CLO 3  row operations and cofactor expansion
    5  Inverses and factorizations    CLO 2  Gauss-Jordan, adjugate, elementary, LU
    6  Rank, kernel and range         CLO 7  the four fundamental subspaces
    7  Vectors and orthogonality      CLO 5  inner products, projections, Gram-Schmidt
    8  Eigenvalues and eigenvectors   CLO 8  spectrum, eigenspaces, diagonalization
    9  Theorem lab                    CLO 1/4/5  verify a statement with a cited certificate
    0  Workbench                      evaluate matrix expressions
    s  Sample problems                37 worked problems across all eight outcomes
    ?  Help and keys
```

## Why exact arithmetic changes the answer

Most tools hand you `0.3333333333333333` and let you decide whether that is a
third. Sylvester never reaches that position. Every number is an exact rational,
and when an eigenvalue needs more than that, it becomes an exact element of a
quadratic field `Q(√d)`. The same field is then used to row reduce `A − λI`, so
the eigenvector comes out exact too:

```
$ sylvester eigen -m "1 1; 1 0"

  det(λI - A) = λ^2 - λ - 1

  λ = (1 - √5)/2   ~ -0.618034    algebraic 1   geometric 1
  λ = (1 + √5)/2   ~  1.618034    algebraic 1   geometric 1

  Eigenspace for λ = (1 + √5)/2
    basis vector v1 = ((1 + √5)/2, 1)
    ✓ check: Av = ((3 + √5)/2, (1 + √5)/2) and λv = ((3 + √5)/2, (1 + √5)/2)
```

`d < 0` is the same machinery, so `[[0, -1], [1, 0]]` reports eigenvalues of
exactly `i` and `-i` rather than a pair of floats near zero.

### When there is no square root to write down

Most characteristic polynomials of a 3×3 or larger matrix do not factor over the
rationals, and their roots have no usable radical form. That does not stop the
answer being exact. Sylvester factors the characteristic polynomial over `Q`
(Zassenhaus: factoring modulo a prime, Hensel lifting, recombination), and each
irreducible factor `q` of degree three or more becomes a *family* of eigenvalues:
the roots of `q`, handled as exact algebraic numbers by computing in
`Q(λ) = Q[λ]/(q(λ))`. Row reducing `A − λI` in that field gives one eigenvector
whose entries are polynomials in `λ`, and because every step is an identity
modulo `q`, it is the eigenvector for every root at once:

```
$ sylvester eigen -m "1 1 0; 1 0 1; 0 1 0"

  det(λI - A) = λ^3 - λ^2 - 2λ + 1
  factored over Q: λ^3 - λ^2 - 2λ + 1   (irreducible)

  λ = each root of λ^3 - λ^2 - 2λ + 1  algebraic 1 each  geometric 1 each    all 3 real

  basis vector v1 = (λ^2 - 1, λ, 1)
  ✓ check: Av = (λ^2 + λ - 1, λ^2, λ)
           λv = (λ^2 + λ - 1, λ^2, λ)
    equal once λ^3 is reduced using λ^3 - λ^2 - 2λ + 1 = 0, so Av = λv for every root
```

How many roots are real is decided exactly with Sturm sequences, not by
inspecting floating point output, and the decimal values printed beside each
root are only there to read. Nothing in the analysis depends on them.

Eigenvalues from different fields are handled the same way. The symmetric matrix
below has eigenvalues `±4√2` and `2 ± 2√5`, so no single square root covers them.
`AP = PD` is still verified column by column, and orthogonality between
eigenvectors from different fields is checked exactly in the tensor ring
`Q[x, y]/(p(x), q(y))`, where every coefficient of the inner product has to
cancel:

```
$ sylvester eigen -m "5 1 3 -1; 1 5 -1 3; 3 -1 -3 1; -1 3 1 -3"

  orthogonality between every pair of columns, checked exactly (6 checks):
    2 by conjugate eigenvalues: the inner product is zero modulo q(x) and (q(y) - q(x))/(y - x)
    4 by different minimal polynomials: every coefficient of the inner product in Q[x, y]/(p(x), q(y)) is zero
```

That check can fail: the test suite deliberately breaks orthogonal eigenbases
and confirms the certificate rejects every one.

## Two ways to row reduce

This is the one idea Sylvester kept from the Gaussian solver it grew out of, and
it is still the most useful thing in it.

**Machine.** Partial pivoting, the way a numerical library does it. Take the
largest entry in the column, scale it to 1, eliminate below. It is correct in
floating point, where dividing by the largest available number keeps rounding
error small. Working exactly, that first division drops a fraction into the
matrix on step one and every later step drags it along.

**Human.** The way a person does it on paper, where the enemy is not rounding
error but arithmetic you have to do in your head.

- **Hunt for a pivot of 1 or −1 first.** Eliminating with a pivot of 1 never
  creates a fraction. This one rule does most of the work.
- **Refuse to divide.** Machine mode scales the pivot row to 1 immediately.
  Human mode leaves it alone and divides once, at the very end, to land the
  leading 1.
- **Scale instead of divide.** Rather than `R2 → R2 − (4/6)R1`, do
  `R2 → 3R2 − 2R1`. Same zero in the same place, no denominator.
- **Cancel common factors.** The moment every entry in a row shares a factor,
  divide it out. `[0, −3, −3, −9]` becomes `[0, −1, −1, −3]`.
- **Manufacture a 1 when none exists.** If two rows differ by exactly 1 in the
  pivot column, subtracting them produces a 1 for free. One extra step to save a
  page of work.

Both reach the same RREF — it is unique, so they must. They differ in how ugly
the scratch work gets:

```
$ sylvester rref -m "6 9 15 3; 4 7 11 5; 8 13 21 9" --compare

                         human  machine
  row operations         11     11
  steps with a fraction  0      9
  worst denominator      1      8
  largest number         21     21
```

Same answer, same effort, and one of them you could have done on paper. The test
suite checks both routes against 1500 randomized matrices and they always agree.
Across 3000 random matrices a fraction appears in about 11% of the human route's
steps against 76% of the machine route's.

## The theorem lab

A computer cannot prove a universally quantified statement by running it. It can
do two useful things instead, and Sylvester is explicit about which one you are
getting.

**Complete proofs.** For a statement that is decidable on the data you gave it —
are these vectors independent, is `b` in this span, is this system consistent,
is `A` invertible — the answer *is* a proof, and it comes with a witness. A
dependent set produces the actual dependency relation, and the relation is then
evaluated to show it really is zero:

```
$ sylvester independence -m "1 3 5; 2 0 4; 2 4 8"

  Linearly dependent.
  Column 3 is free, which produces an explicit dependency:
    (-2)v1 + (-1)v2 + (1)v3  =  0
  Not every coefficient is zero, so the set is dependent.
  ✓ checked: the combination really is the zero vector.
```

**Verified instances.** For a general identity such as `det(AB) = det(A)det(B)`,
running it on your matrices does not prove the theorem. Sylvester computes both
sides, shows them, states that they agree, and cites the theorem that covers the
general case. The certificate says which of the two you are looking at.

39 propositions, grouped the way the course groups them:

| Group | Includes |
|---|---|
| Systems of linear equations | consistency by rank, wide homogeneous systems, particular + homogeneous structure, superposition, uniqueness against the null space |
| Matrices and determinants | `(AB)ᵀ = BᵀAᵀ`, `(AB)⁻¹ = B⁻¹A⁻¹`, `det(AB)`, `det(Aᵀ)`, `det(cA)`, `det(A⁻¹)`, the adjugate identity, row operations on determinants, triangular determinants, the Invertible Matrix Theorem, `tr(AB) = tr(BA)`, symmetric/skew splitting |
| Vectors, subspaces and orthogonality | Cauchy–Schwarz, triangle, parallelogram, Pythagoras, projection orthogonality, orthogonal ⇒ independent, independence, span membership, the subspace test, rank–nullity, row rank = column rank, Gram–Schmidt preserving span, complement dimension, extracting a basis |
| Eigenvalues and eigenvectors | the eigenvector definition, trace and determinant from the spectrum, independence for distinct eigenvalues, the spectral theorem, Cayley–Hamilton, similarity invariants |

Every one is checked against randomized operands in the test suite, so a
proposition that is stated wrongly fails CI rather than quietly reporting
"verified".

## Course coverage

Built to cover the SJSU linear algebra course outcomes. `sylvester outcomes`
prints this mapping with the sample problems for each.

| Outcome | Where |
|---|---|
| 1. Prove elementary statements about systems of linear equations | `solve`, theorem lab (systems group) |
| 2. Elementary row operations, inverses and transposes | `rref`, `inverse` — Gauss-Jordan, adjugate, elementary factorization, LU |
| 3. Determinants by row operations and cofactor expansion | `det --method rowops`, `det --method cofactor` |
| 4. Prove elementary statements about matrices and determinants | theorem lab (matrices group), the Invertible Matrix Theorem |
| 5. Vectors, inner products, projections, norms, orthogonality, independence, spanning sets, subspaces, bases, dimension, rank | `vectors`, `gram-schmidt`, `project`, `span`, `basis`, `independence` |
| 6. Determinants to solve homogeneous and non-homogeneous systems | `solve --method cramer`, `solve --method homogeneous` |
| 7. Rank for independence, kernel, range and nullity | `subspaces` — all four fundamental subspaces |
| 8. Eigenvalues, eigenvectors and eigenspaces | `eigen` — characteristic polynomial factored over Q, exact eigenvalues of any degree, eigenspaces, diagonalization, spectral theorem |

## Install

Python 3.9 or newer. Nothing to install.

```bash
git clone https://github.com/DevomB/Sylvester
cd Sylvester
python -m sylvester
```

Optionally put the `sylvester` command on your PATH:

```bash
pip install -e .
```

## Terminal interface

`python -m sylvester` opens the full-screen interface. It is written directly
against ANSI escapes with raw key input through `msvcrt` on Windows and
`termios` elsewhere, so it runs anywhere Python does with nothing installed.

| Key | Does |
|---|---|
| `↑ ↓` | move a selection, or scroll a report |
| `← →` | pan a wide report sideways |
| `PgUp` `PgDn` `Home` `End` | scroll faster |
| `enter` | open the highlighted item |
| `r` / `b` | cycle which register the report uses |
| `v` | cycle the method or view |
| `m` / `s` / `a` / `c` | reduction strategy, steps, augmented column, compare |
| `esc` / `q` / `?` | back, quit, help |

Matrices live in registers `A`–`Z` and are shared across every screen, so you
can edit `A` once and then reduce it, invert it, take its determinant and find
its eigenvalues without retyping it. In the editor, arrows move, typing edits,
`+` `_` add and remove rows, `]` `[` add and remove columns, `t` transposes, `i`
makes an identity, and `w` writes the register.

The workbench evaluates expressions against those registers:

```
> A*B^T + 2*I(3)
> inv(A)*A
> det(A*B) - det(A)*det(B)
```

## Command line

Every screen has a scriptable equivalent that writes to stdout.

```bash
sylvester rref     -m "2 -1 3; 1 1 -2; 4 1 -1" --compare
sylvester solve    -m "1 2 3; 2 4 8; 3 6 11" -b "4 10 14"
sylvester solve    -m "1 2 | 3; 4 5 | 6"              # augmented form
sylvester det      -m "1 2 0; 3 0 4; 5 6 0" --method cofactor
sylvester inverse  -m "1 2 3; 0 1 4; 5 6 0" --method elementary
sylvester subspaces -m "1 2 3 4; 2 4 7 10; 3 6 10 14"
sylvester eigen    -m "5 4 2; 4 5 2; 2 2 2"
sylvester vectors  -u "1 2 2" -v "3 0 4"
sylvester prove    det-product "2 1; 3 4" "1 -2; 5 0"
sylvester eval     "det(A)" -m "A=1 2; 3 4"
sylvester samples                      # 37 worked problems
sylvester outcomes                     # course outcomes and coverage
```

Entries may be integers, fractions like `-2/3`, decimals like `1.5`, or
`sqrt(8)`. Rows are separated by `;`, values by spaces or commas. `--ascii`
drops the box drawing for terminals that cannot render it, and `--no-color`
disables ANSI color.

## As a library

```python
from fractions import Fraction
from sylvester import Matrix, Poly, factor, four_subspaces, prove, solve, spectrum

A = Matrix([[1, 1], [1, 0]])
spec = spectrum(A)
spec.pairs[1].value           # exact (1 + sqrt 5)/2
spec.diagonalizable           # True

cubic = spectrum(Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 0]])).pairs[0]
cubic.minimal                 # x^3 - x^2 - 2x + 1, irreducible over Q
cubic.basis                   # [(λ^2 - 1, λ, 1)] with λ an exact algebraic number
cubic.real_count              # 3, decided by Sturm sequences

factor(Poly([4, 0, 0, 0, 1])) # (1, [(x^2 - 2x + 2, 1), (x^2 + 2x + 2, 1)])

solution = solve(Matrix([[1, 2, 3], [2, 4, 8]]), [Fraction(4), Fraction(10)])
solution.kind                 # 'infinite'
solution.particular           # a particular solution
solution.homogeneous_basis    # a basis for the null space
solution.verify()             # True, checked exactly

four_subspaces(A).rank
prove("det-product", A, A).holds
```

## Tests

```bash
python -m unittest discover -s tests -t .
```

204 tests, mostly randomized property checks: that both reduction strategies
reach the same RREF, that four independent determinant algorithms agree, that
`A·adj(A) = det(A)I` and `p_A(A) = 0`, that every null space vector is
orthogonal to every row space vector, that no proposition in the theorem lab can
be made to fail, and that the terminal interface renders every screen without a
line ever running past the edge.

The algebraic side is tested against ground truth rather than against itself.
Products of Eisenstein polynomials, which are provably irreducible, must factor
back into exactly those pieces; `x^4 + 1`, which is irreducible yet splits modulo
every prime, must survive recombination intact; every eigenvector of an
irreducible family is evaluated at each numeric root and checked against `Av`;
and every exact spectrum on random matrices up to 6×6 must verify, with its
trace and determinant matching.

## License

MIT

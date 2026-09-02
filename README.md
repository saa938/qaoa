## Usage

Layout
```
  src/      all scripts (run these; they resolve data/results/figures relative to their own file location)
  data/     input data (MaxCutMAQAOAData.csv, graphs.csv, AshayMAQAOAData/)
  results/  generated json/csv, and results/shells/ for the cached .npz minima
  figures/  generated png plots
```
Scripts can be run from any working directory, e.g. either of:
```
  python src/verify_all.py
  cd src && python verify_all.py
```
## Learnings

### A. Conventions and bugs

**Period is pi, not 2pi:** Adding pi to any angle gives the exact same energy, leading to double-counting

**E(x) = E(-x):** If you flip the sign of every angle, the energy stays the same. Minima always come in mirror image pairs.

**Ian's gamma vectors use node-insertion ordering:** Also important to check graph construction ordering (NetworkX vs sorted)

### B. Landscape Structure

**Minima keep on growing:** Ran 10,000 random restarts, and the count of new minima kept on going up at a steady pace. When we stopped, it was 4,403 distinct minima.

**Bimodal curvature:** Looking at the curvature 54% had near-zero values and 46% had values around 2. There are also two kidns of minma: proper bowls and flat troughs.

**Troughs are real:** Maybe there are no troughs and it is just tightly packed bowls. We walked a distance of 1.0 along the supposed trough and the highest energy bump was 3 * 10^-13, proving it is a trough.

**No one target per graph::** There are hundreds of variations of optimal angles confusing the GNN.

**Centroid doesn't mean anything:** The average of all the minima is essentially at the origin. This means they are scattered uniformly in all directions and cancel out. 

**No crystal structure:** Pairwise distances between minima don't cluster at a few specific values, meaning it is a broad smooth distribution.

### C. GNN Findings

**The GNN lands on basin boundaries:** Starting the optimizer at the GNN's prediction, only 28% of the runs reach the true floor. If you move it slightly in a small, random direction more than half of the runs reach the true floor.

**GNN error tilts toward flat directions:** About 53% of the error magnitude sits in low-curvature directions, versus ~26% if the error was random. 

### D. Penalty Methodology

**Distance squared is better than distance:** When trying to eliminate solutions far away from the origin, distance squared works a lot better. Decreased distinct minima from 84 to 9-13 and raised the rate of reaching the floor from 42% to 75%.

**Multiplicative penalties must be floor-zeroed:** If the energy is negative and you multiply by a factor, the larger the factor is the more negative the energy becomes, which is the opposite of what is needed. Instead, shift the floor so it is zero.

**One penalty strength doesn't fit all:** One penalty strength that gently nudges a graph shoves another to a wrong basin. Solution: run multiple values, including 0 to find the real minima of the true energy.

**Don't estimate the floor separately:** Take the floor as the minimum over the entire harvest to ensure all the minima are correct.

### E. Quantization and shell geometry

**pi/4 quantization holds on 4/10 graphs**: In graphs 11, 13, 15, and 17, every angle in the lowest shell sits on an exact multiple of pi/4. 

**Quantization means no flat directions:** The four graphs with pi/4 quantization are the same four whose minima are isolated points.

**Points aren't distributed equally:** The nearest neighbor of each minima is the same distance away on only 2/10 graphs. Mostly, it is irregular.

**Quantization doesn't work for p > 1:** At p=2 and p=3 the quantization property completely goes away.

**Flat directions increase significantly with p:** In graph 13, there are 0 flat directions at p=1, 12 at p=2, and 30 at p=3. This also explains the point above since flat valleys can't fit into a grid.

**The radius is explained by coordinate compostion:** The squared shell radius (in units of (pi/4)^2) lands on a specific quantized value per graph. 

**Edge count does not predict radius:** You would think bigger graph means bigger radius. Across m=13 to m=21 the radius barely moves (2.616-2.939) and at both m=14 and m=17 two different graphs with the same edge count have different radii.

**Radius doesn't grow with layers:** Going from p=1 to p=2 doubles the number of parameters, but it doesn't change the radius.

**Graphs 14 and 19 are weird:** They have lots of flat directions even at p=1, heavily degenerate curvature, and optimization on them is unreliable. 

### F. Floor value structure

**The floor is an exact multiple of 0.5 98% of the time:** The floor is always something like -11.0, -12.5, -9.5, except for one case where it was -11.077.

**The counterexample:** A bit more about the graph with the different floor. It had 9 nodes and 16 edges, and a max cut of 13. Also, −10.5 − 1/sqrt3 = −11.077. This was verified through 1000+ restarts

**1/sqrt3:** This usually appears as the offset on non-floor local minima on almost every graph I tested. Except on this graph it actually was the floor.

**p=1 has a gap, p=2 reaches the max cut:** On half of the 10 graphs the first iteration doesn't reach the max cut value, but on the second one it does.

### G. Radius based search

**Capped cost function is successful:** Outside of the given radius, set the gradient to 0 so the optimizer stays away. Find a solution, lower the bound to its radius, and repeat, until you reach the true shell.

**Hit rate decreases, need to run multiple iterations:** As the radius decreases, it becomes harder to find new minima: 15/80, then 5/80, then 1/80. Requires 2-3 consecutive stalled iterations before stopping.

**Shaping helps and exp(0.5) is the best:** With no shaping at all, the search never reaches the shell on any graph. Which shaping wins is graph-dependent, but exp(0.5) is best on 5 of 10 and produced the single best result anywhere (graph 11: shell reached on 37 of 40 restarts, 286 evaluations per hit). power(3.0) is the safest: non-zero on 7 of 10. exp(1.5) is best only on graph 17 and found nothing at all on graph 15, so it's a bad default despite looking good on a two-graph sample early on.

**Positive fraction method helps:** Using the fraction of coordinates that are positive roughly halves the work: graph 13 went 88196 to 39715 evaluations for example.

**The multiplcative version is terrible:** Multiplying the cost by exp(1 − fraction) blew graph 13 from 23510 to 601656 evaluations. This is 25 times worse.

**However, positive fraction doesn't yield a unique answer:** Because of energy symmetry, it can't distinguish between the two and can't yield a unique answer.

### H. Harvest Data

**Floors match, but shell radius is larger on graphs 11 and 12:** Both agree on floors, but harvest didn't find the true shell because of sampling.

**Weighted edges:** With weighted edges, radii ranged from 1.993-3.061. The 1.993 sits well below the rest, might need to check it again.

**p=2 needs way more restarts:** Double the parameters means more places to get stuck, so it requires significantly more restarts.

### I. Degeneracy Results

**Exhaustive sign enumeration replaces restarts:** On 8/10 graphs every shell coordinate lands on the pi/8 grid. Instead of hoping restarts find every point, enumerate all sign patterns at each observed magnitude and keep the ones on the floor.

**The shell is almost entirely sign flips:** Most of the shell is one or a few magnitude patterns with many admissible sign patterns. On graph 15 all 256 points have identical coordinate magnitudes and differ only in signs.

**Radius and weighted radius are blind to signs:** Both only depend on the coordinates squared, so they take the same value at every sign flip. On the eight exact graphs the weighted radius takes only 1, 2, or 6 distinct values across shells of 64 to 512 points.

**Radius then weighted radius does not work:** Leaves 64 or more points on 8/10 graphs, and on graphs 11, 12, 13, 14 and 15 it removes nothing at all from the shell. This is structural, so no choice of weights fixes it. Adding positive fraction helps, but isn't enough.

**The weight set barely matters:** Linear (1,2,3,...), powers of 2, and primes give the same survivor counts everywhere except graph 18.

**Weighted positive radius always gives exactly one point:** A coordinate only contributes its weight when it is positive. This picks 1 point on all ten graphs. However, it depends on the edge list order instead of the graph.

**Odd graph invariants work:** Odd is important to keep the sign. Score with sum_gamma, sum_beta, deg_gamma, deg_beta, prod_gamma and tri_gamma, each linear and cubed, applied in a fixed order keeping the argmax set at each step. Gives one point on 8/10 graphs.

**Only three of the twelve functionals ever fire:** sum_gamma cuts on all ten graphs, deg_gamma on graphs 10, 16 and 17, deg_beta on graph 18. prod_gamma, tri_gamma, sum_beta and every cubic version never cut anything.

**Automorphism groups are the floor:** Graphs 15 and 18 leave 2 and 4 points respectively, which is exactly one group each. No function of the graph structure can separate points related by graph symmetry. All 10 graphs reach one automorphism group.

### J. Invariants that ignore parameter position

**Sign Counts are spread out:** If every parameter has the same magnitude but only signs differ, then measures that ignore position would give almost identical values to solutions with the same number of pluses and minuses. However, the most common value only accounts for 23-31% of the shell. Solutions with an equal number of pluses and minuses are 0% of the shell on all eight.

**Splitting makes it work better:** Treating all parameters as one "block", the smallest group is 8 to 64 points depending on the graph. Scoring the cost parameters and mixer parameters as separate helps, most of them are 1-2 but graph 18 is 8. It is a little better, but not as good as ```invariants.py```.

**Only the plain gamma sum does stuff:** ```sum_gamma``` is the only measure that removes any points, on every graph. Standard deviation, e_2, e_3, e_4, positive fraction and every beta-block measure never once change the outcome. 

**Energy symmetry makes standard deviation suboptimal:** Standard deviation gives the same value for x and -x, and so does every even-order elementary symmetric polynomial. Because the shell always contains both a solution and its mirror image, no combination of even measures can get below 2 points. 
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

### K. More layers and weights

**Periodicity changes:** For weighted edges, the energy is not pi periodic in gamma, the period is pi divided by the weight on that edge. The cost unitary is exp(i gamma_e w_e Z_u Z_v), so shifting gamma_e by pi / w_e multiplies every amplitude by -1, a global phase. Beta stays pi periodic.

**Sign flip still exists:** Even with weights and more layers E(x) = E(-x) still holds. A shell can never hold fewer than 2 points, and if it is 2 it is just the point and the mirror.

**Random weights destroy graph symmetries:** If the edges are weighted then they are different from each other, allowing the filter to return one point on all 20 trials.

### L. Global vs local optimization

**minimize beats basinhopping:** Measuring evaluations spent per global minimum found, plain L-BFGS-B restarts are cheaper on 13 of 16 cases. Median ratio 1.24, geometric mean 1.54, worst case 3.96x.On the 3 cases basinhopping was better, the number of basins used before hitting was exactly 1, meaning it found the floor in its first descent and did what a single minimize call does. 

**The default stepsize is too small:** Energy has period pi and scipy's default perturbation is 0.5, so it keeps falling back into the same basin. At stepsize pi/2 the median ratio drops to 0.97 and it ties with restarts. It still never beats them.

**Density alone doesn't predict difficulty:** At p=1 the three graphs that don't reach the max cut have hit rates 0.63, 0.97 and 0.575, all above the three that do (0.505, 0.08, 0.18), regardless of density. Whether the floor equals the max cut matters more than how many edges there are.

**Capping the radius costs monotonically more:** On g8_d50 at p=1, evaluations per success go 5622, 11569, 28998, 81305 as the cap goes 1.4, 1.2, 1.1, 1.0 times the shell radius. That is 14 times worse for a cap only 40% tighter.

**P=1 is only one where minimze performs better:** The ratio of the cost of basinhopping over the cost of minimize for p=1 was 2.97 (cheaper on all). By p=2 the ratio drops to 1.17, and by p=3 the two methods are statistically indistinguishable at 0.97.

**Weighted p=1 is harder to find, not harder to reach:** Median per-attempt success falls from 0.99 unweighted to 0.51 weighted, but the quality of the p=1 answer barely moves: median approximation ratio 0.9296 unweighted against 0.9346 weighted. Weighting changes how hard the optimum is to find, not how good p=1 is.

### M. Characteristics of weighted graphs

**The pi/4 shell quantization still exists:** shell_r^2 / (pi/4)^2 is an integer in 21 of 24 weighted p=1 cells, on values 8, 10, 12 and 16, against 10 of 17 unweighted. It breaks at p >= 2 for both, consistent with the quantization being a p=1 property seen for unweighted graphs.

**Grid also scales with weight:** For period, which was pi and became pi/weight for each edge angle, the grid also scales simiarly. It becomes pi/(constant*weight). 

### N. Half-period symmetries

**Shifting a gamma by half a period is a symmetry:** Add pi/(2w_e) to gamma_e in layer l and flip the signs of beta_u and beta_v in layer l and every later layer, and the energy stays exactly the same. Checked at p=1, 2 and 3, weighted and unweighted, to 1.8 * 10^-15, while the shift alone changes the energy by order 1.

**Shifting a beta by half a period is also a symmetry:** Add pi/2 to beta_j in layer l and flip the signs of the gammas on j's edges in layer l and every earlier layer. Checked to 1.8 * 10^-15 in the same cases.

**Old shells were mostly copies:** Unweighted p=1 shells of 64 to 552 points shrink to 1 to 12 points. On weighted p=1, 14 of 16 cases have one point before any function is applied, and the other 2 only need sum_gamma. The 98 points on graph 16 (weight draw 1) were mostly copies.

**A shell can now be one point:** Section K says a shell never has fewer than 2 points because of the mirror image. After reducing, the mirror of a point can reduce back to itself. This happens on unweighted graphs 11, 12 and 13 and on most weighted cases.

### O. Flat valleys and reproducibility

**Points in troughs weren't reproducible:** On a trough every restart lands at a different spot, so the pick depended on sampling. On graph 16 (weight draw 1) at p=1 two seeds picked points 4.9 * 10^-3 apart. At p=2 on graph 16 (weight draw 0) the two picks were 2.94 apart with different radii.

**Traverse each point to the closest spot on its valley:** Step toward the origin along the directions where the curvature is zero, polish, reduce, and repeat. With this, two seeds on graph 16 (weight draw 1) agree to 7 * 10^-8.

**Curved valleys need a penalty:** When the valley bends, those steps overshoot and the walk stops short. Instead, pull the point toward the origin with a penalty on the energy   keeps it on the floor, making the penalty stronger in steps (10 up to 10^7), then do one more straight step to tighten. This doesn't need to know which way the valley bends. 

**Sign flips that stay on the floor aren't always symmetries:** Some points at the same radius differ only in the signs of some coordinates. If the flipped coordinates sit on the edge, it is one of the symmetries in N. On graph 16 most pairs flip coordinates in the middle of their range, and the same flips change the energy by order 1 at random points, so they are different minima. Test a flip on random points before counting two points as one.

### P. Efficiency and number of restarts

**Parameter shift was the bottleneck:** It costs 2 energy evaluations per parameter. The adjoint gradient runs the circuit forward once and backward once and reads every derivative off on the way back, the same idea as backpropagation. It matches parameter shift to 9.4 * 10^-15, and polishing is 4.9x faster at p=1, 9.2x at p=2 and 15.6x at p=3.

**Restarts vs basinhopping:** Measuring evaluations until the chosen solution is found on 16 weighted p=1 cases, the restart mix is cheaper on 8, basinhopping with stepsize pi/2 is cheaper on 4, and 4 are within 25%. No hit rate cutoff separates them: graph 17 (weight draw 1) has a 3.7% hit rate and restarts are still 3.7x cheaper.

**One penalty still doesn't fit all:** Lambda 0.35 lands on the answer in 93 of 100 restarts on graph 10 (weight draw 0) against 7 with no penalty, but 0 of 100 on graphs 11, 14 and 16 (weight draw 0). Cycling through 0, 0.1 and 0.35 covers both.

**Where restarts start doesn't matter:** Seeding in [0, pi) and seeding over each coordinate's full period hit the answer 110 times each.

**How many restarts:** If one restart finds the answer with probability q, the chance that N restarts all miss it is (1-q)^N. Setting that to 1% gives N = ln(100)/q, about 4.6/q. 
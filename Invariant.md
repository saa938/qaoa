# Symmetries of a graph, and how to use them

## Automorphism symmetry
If you rename some vertices of a graph, the edge list can still be the same (automorphism)

Here are the counts for the 10 Erdos-Renyi graphs:
 graph | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|-----|----|----|----|----|----|----|----|----|----|----|
| number of automorphisms | 2 | 12 | 2 | 1 | 1 | 4 | 1 | 2 | 8 | 12 |

Renaming nodes changes the parameter vector, but the energy is still the exact same. There is one beta per node and one gamma per edge. If the relabelling sends node 0 to node 1, then whatever value beta was carrying for node 0 now belongs to node 1. 

Take a solution, and if you swap two nodes it will move, but it is still the same solution. So even though there are a lot of solutions, in reality most of them are duplicates.

## Energy symmetry

Since E(x) = E(-x), flipping the sign of every angle gives another point with the same energy. Even though they are distinct points, they can be grouped as one. 

## Grouping

We use a function called  ```canonicalize()``` where given a point, it generates every version of it (all automorphisms with energy symmetry) and returns the lexicographically smallest. Running this over the minima in the shell splits it into groups. Different groups are actually different solutions. 

## Choosing the best group

We still have multiple groups, so in ```invariant.py``` we apply a list of scoring functions in a fixed order. The scoring functions were created such that it is based on the entire graph instead of a particular ndoe or edge. These are the scoring functions

 - ```sum_gamma```: every gamma coordinate weighted 1 (a plain sum)
 - ```sum_beta```: every beta coordinate weighted 1
 - ```deg_gamma```: each gamma weighted by the sum of its edge's two endpoint degrees
 - ```deg_beta```: each beta weighted by that node's degree
 - ```prod_gamma```: each gamma weighted by the product of its edge's endpoint degrees
 - ```tri_gamma```: each gamma weighted by how many triangles that edge sits in

Each one also has a cubed version, where the weight multiplies the cube of the parameter instead of the parameter itself, giving twelve in total. We can't use quadratic because it doesn't account for negatives.

## Results
 
| graph | shell size | automorphisms | groups | points left after the filter | all one group? |
|-----|-----------|---------------|--------|------------------------------|----------------|
| 10 | 128 | 2 | 48 | 1 | yes |
| 11 | 128 | 12 | 24 | 1 | yes |
| 12 | 64 | 2 | 32 | 1 | yes |
| 13 | 64 | 1 | 32 | 1 | yes |
| 14 | 38 | 1 | 33 | 1 | yes |
| 15 | 256 | 4 | 40 | 2 | yes |
| 16 | 128 | 1 | 64 | 1 | yes |
| 17 | 128 | 2 | 64 | 1 | yes |
| 18 | 512 | 8 | 44 | 4 | yes |
| 19 | 158 | 12 | 108 | 1 | yes |

All of them end up in one point except for two graphs, but all of them end up in one group. Note that graphs 14 and 19 are the weird ones so it is a lower bound instead of a exact count, because those two graphs are the ones whose minima do not land on the pi/8 grid, so we cannot enumerate their shells exhaustively.
# python-algorithms
Useful algorithms implemented in python.

---

This repository is a collection of:

 * useful algorithms implemented in python using the standard library only;
 * sample applications used to demonstrate the use of important algorithms existing in the standard lib.

This is mostly intended as a self study and reference material in my journey to learn Python
and not as production ready code; the test are also much more limited than I'd like to have in a real project.

I'd love if this can be useful to someone out there too, so please help yourself.
If you like it or, even more likely, if you find parts (or all) not being "pythonic enough" please let me know!
All of this is done to learn, so do not hesitate commenting.

## Sort
The sort module implements some sorting algorithms, using basic and (hopefully) clear implementations.
At the moment there are these sort algorithms:

* **insertion sort**: an in place, stable, no extra memory algorithm with quadratic performance in general,
but very fast for short (less than 10-15 elements) or partially sorted lists.
* **merge sort**: an in place, stable, double memory usage algorithm with linearitmic (~ N * log N) performance in general.

## kdTree
The module implements a K dimension set based on a K dimension Tree.

This is an ordered tree where multidimensional points are stored in order according to one different dimension
at every level of the tree, actually allowing very fast (linearitmic) operations for search and more complex queries
like **nearest neighbour** and **range** queries.   

The current implementation is (lightly) tested for K = 2 and K = 3, so for 2D and 3D points,
but should work equally for points with K dimension.

The module could also be extended to implement a Symbol Table to associate to a K dimensional point some value.

## 8 puzzle solver
The module implemnts a solver for the 8/15/N puzzle game, using the A* algoritm (based on priority queue).

### 8 puzzle game
The game consists of a square board of size N * N where are placed tiles numbered from 1 to (N^2 -1),
so that it left an empty square.

The goal is to rearrange the tiles in numerical order with 1 in the top left corner and the empty tile in the lower
right, with the less possible moves.

Every move consists of sliding a tile in the currently empty square by moving it horizontally or vertically.
The actual result is to exchange position with the empty square. Naturally only tiles adjacent to the empty square can
be moved at a certain time.

### A* algorithm
The A* algorithm is implemented using the Python supplied priority queue module.

The basic idea is to define a distance function (called Heuristic) that can measure how "far" is the current board from
the goal and then follow primarily the moves that reduce this distance.

To be able to compare the "goodness" of two next steps even if they are not at the same number of moves from the
beginning the move number of a node is added to his distance from the solution to form a priority function that can
account both for the moves already done and the ones to still (expected) to be done.

At every move we then perform the candidate move with the lowest priority and if that's not the goal we add in the
priority queue the possible moves from this board state (excluding the one that reverses the move just done).
The maximum number of moves added at every step is three, but they can be just one if the empty square is in a corner.

An important property of the A* algorithm and of the Heuristics used is that if the Heuristic is admissible and
consistent then the priority of candidate moves taken from the priority queue never decrease and the priority of
candidate moves added to the priority queue is never less than that of the move they are generated from.
NB A heuristic is both admissible if never overestimates the number of moves to the goal and it is consistent if
satisfies a certain triangle inequality.

### Priority Queue
The A* algorithm is based on the property of Priority Queues to be able to add and delete a huge number of elements
in a very efficient way while keeping them sorted on their priority values.
This allows to enqueue all possible moves and follow the most promising ones in a very simple, natural and global way,
without having to bother too much of how many moves are enqueued or to decide at every step the best direction to go
based on limited "local" knowledge.


---

   All the algorithm provided in this repository are Copyright 2016 by Roberto Zagni

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.

   A copy of the License is in the LICENSE file in this repository or you may obtain it at http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

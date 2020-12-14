
## File Overview

./bin - top-level scripts
./data - supplemental data to the Waterloo dataset
./fractal - Core implementation
./tests - miscellaneous testing scripts
./Results - miscellaneous results

## Project Overview

Fractal Image compression is a technique that attempts to learn a mappingTthat can be iteratively applied to anysource signal such that the output will converge on the encoded image, called anattractor; this requires capturingrecursive structures in images in a computationally efficient way.  In the majority of the Fractal Compressionliterature, the operator T is taken to be an affine transform, and the encoding process to determine  T is on the order O(n^2). In this report, we present a new technique, based on singular value decomposition, that achieves acomputational complexity of O(n) and produces results that are comparable to traditional fractal compressionschemes as well as more conventional compression algorithms such as JPEG

## Authorship
Vrishab Commuri 
Scott Mionis
Bridget Tan
Weishan Wang

./fractal - Vrishab and Scott 
./fractal/utils - Bridget and Weishan
./bin - Bridget and Vrishab
./docs - Scott

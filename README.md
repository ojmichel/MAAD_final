# Description

This is my final project for Media Art Design and Practice. Inspired by cellular automata, I wanted to build something where structure emerges over time in a simulation governed by mathematical rules. The problem with these simulations however is that they mostly rely on hand-crafted initialization states for interesting patterns to emerge, and they are entirely deterministic. I wanted to make something where patterns would consitently emerge from nothing, and that would also evolve with some randomness. To do this, I used the following equations to update the value of a pixel based off the value of it's neighbors:

The second equation embeddeds the pixel vector into a sphere which forces pixels to interact with each other (they can’t avoid interaction by going off to infinity).

Here is an example run:

For some cool examples try running:

`python3 main.py --W 25 --H 25 --mode mix --sigma 10.0  --rad 1.0 --s demo3.mp4`

`python3 main.py --W 100 --H 100 --mode mix --sigma 10.0  --rad 1.0 --f image100.jpeg`

`python3 main.py --W 50 --H 50 --mode mix --sigma 5.0  --rad 1.0 --f image100.jpeg`

`python3 main.py --W 25 --H 25 --mode bar --sigma 10.0  --rad 1.0 --init zero --sigma_init 1.0 --s demo1.mp4`

`python3 main.py --W 50 --H 50 --mode bar --sigma 10.0  --rad 1.0 --init rand --sigma_init 1.0 --s demo2.mp4`

`python3 main.py --W 25 --H 25 --mode bar --sigma 10.0  --rad 1.0 --f image100.jpeg --s demo5.mp4`

#Photography case

##Aim

1. Taking pair of frames (impulse response h and the image) check h(x,y)(z) dependence
where z is defocus distance and h(x,y) can be treatead as "shape of defocues dot"
//Can we just use crossection width of the h(x,y) ?

2. Determine possible problems for more complicated case in particular:
    - check quality of the "ideal dot" in focus case
    - noise?

3. find optimal and repeatable conditions of recording shots

4. try method of capturuing one shot having both h and image at one frame


##Setup

incoherent light source (X) (divergent wave)
lens for making plane wave (L)
lens with ouput focal lenght f (M)
2D object (O)
confocal lens and photosensitive matrix inside camera (C)

C should have small DoF
C at initial position should capture idally short image

def display_dot_on_object(d):
"""d - distance between LL and O"""

    X______LM____O
    _________C


def enligth_whole_object(d):
"""d - distance between LL and O"""

    X______L_____O
    _________C


##Experimental method


for d in [f, f+dx, ... f+n*dx]:
    1. make_environment_dark()
    2. display_dot_on_object(d)
    3. take_shot()
    4. enlight_whole_object(d)
    5. take_shot()

So we have sht like:

    X_____LM___O
    ________C

    X_____L____O
    ________C

    X-----LM-----O
    ________C

    X-----L------O
    ________C

    X-----LM-------O
    ________C

    X-----L--------O
    ________C
    ...

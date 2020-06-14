import turtle as t

def square(turtle_name, length):
    for i in range(4):
        turtle_name.fd(length)
        turtle_name.lt(90)

def polygon(turtle_name, n, length):
    angle = 360/n
    for i in range(n):
        turtle_name.fd(length)
        turtle_name.lt(angle)

def circle(turtle_name, r):
    circum = 2 * 3.1415926 * r
    n = int(circum/3) + 3
    length = circum / n
    polygon(turtle_name, n, length)

def arc(turtle_name, r, angle):
    arc_length = 2 * 3.1415926 * r * (angle/360)
    n = int(arc_length/3) + 3
    step_length = arc_length/n
    step_angle = angle/n

    turtle_name.color('black', 'yellow')
    turtle_name.begin_fill()

    for i in range(n):
        turtle_name.fd(step_length)
        turtle_name.lt(step_angle)

    turtle_name.end_fill()

def petal(turtle_name, r, angle):
    for i in range(2):
        arc(turtle_name, r, angle)
        turtle_name.lt(180-angle)

def flower(turtle_name, n, r, angle):
    for i in range(n):
        petal(turtle_name, r, angle)
        turtle_name.lt(360.0/n)

mike = t.Turtle()
flower(mike, 10, 300, 45)
t.mainloop() # Main Loop

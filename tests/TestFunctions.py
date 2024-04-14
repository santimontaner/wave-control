import numpy as np

def p(x,y):
    return 2*x**2-y**2+2*x*y+y+x

def px(x,y):
    return 4*x+2*y+1

def py(x,y):
    return -2*y+2*x+1

def p1(x,y):
    return 2*x**3-y**3+2*x*y+y+x

def p1x(x,y):
    return 6*x**2 + 2*y + 1

def p1y(x,y):
    return 2*x - 3*y**2 + 1

def p2(x,y):
    return 2*x-y+1

def p2x(x,y):
    return 2

def p2y(x,y):
    return -1

def p3(x,y):
    return x*y-y*x**2

def p3x(x,y):
    return y-2*x*y

def p3y(x,y):
    return x-x**2

def p4(x,y):
    return np.sin(np.pi*x)*(x**2+y**2)

def p4x(x,y):
    return np.pi*np.cos(np.pi*x)*(x**2+y**2)+2*x*np.sin(np.pi*x)

def p4y(x,y):
    return 2*y*np.sin(np.pi*x)

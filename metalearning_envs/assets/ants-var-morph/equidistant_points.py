import numpy as np
def gen_points(num_points):
    r = 0.2828
    all_points = [(r*np.cos(theta), r*np.sin(theta)) for theta in np.linspace(0, 2*np.pi, num_points+1)[:num_points]]
    print(all_points)

gen_points(3)




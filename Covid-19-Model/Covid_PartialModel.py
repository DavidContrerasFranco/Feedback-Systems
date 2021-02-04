import numpy as np
from numpy import pi
from scipy import arange
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from control.phaseplot import phase_plot
from control import *
from control.matlab import *
from control import ss2tf

N = 49648685

# Beta : 1.94 Medido en China, 1.14 Estudios
Real_S = np.array([49648683, 49648682, 49648677, 49648671, 49648666, 49648663, 49648659, 49648650, 49648633, 49648620, \
                   49648606, 49648584, 49648539, 49648500, 49648450, 49648384, 49648283, 49648136, 49647994, 49647871, \
                   49647727, 49647573, 49647387, 49647230, 49647096, 49646956, 49646831, 49646683, 49646532, 49646373, \
                   49646231, 49646130, 49645996, 49645896, 49645737, 49645615, 49645493, 49645366, 49645239, 49645090, \
                   49644961, 49644840, 49644697, 49644540, 49644397, 49644185, 49644037])
Real_C = np.array([   2,    3,    8,   14,   19,   22,   26,   35,   51,   64, \
                     78,   98,  143,  176,  221,  282,  365,  495,  624,  722, \
                    854,  982, 1123, 1247, 1352, 1422, 1444, 1528, 1668, 1779, \
                   1852, 1861, 1903, 1897, 1904, 1928, 1940, 1923, 1969, 2023, \
                   2021, 1811, 1788, 1704, 1629, 1766, 1832])
Real_I_C = np.array([   0,    0,    0,    0,    0,    0,    0,    0,    1,    1, \
                        1,    3,    3,    9,   14,   18,   35,   51,   63,   87, \
                       99,  124,  168,  196,  221,  279,  368,  415,  408,  434, \
                      465,  523,  571,  638,  744,  798,  849,  929,  976, 1031, \
                     1120, 1385, 1484, 1663, 1813, 1799, 1790])
Real_I_H = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, \
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0, \
                      0,  0,  0,  0,  0,  0,  2,  2,  2,  3, \
                      5,  7, 10, 11, 18, 22, 29, 33, 37, 40, \
                     44, 55, 64, 71, 85, 87, 90])
Real_I_U = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, \
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0, \
                      0,  0,  0,  1,  1,  3,  3,  3,  3,  3, \
                      3,  5,  6, 10, 12, 15, 19, 23, 27, 30, \
                     36, 46, 53, 59, 70, 72, 75])
Real_R = np.array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, \
                     0,   0,   0,   0,   0,   1,   2,   3,   4,   5, \
                     5,   6,   7,  11,  15,  25,  37,  54,  72,  93, \
                   129, 159, 199, 233, 270, 307, 355, 411, 437, 471, \
                   503, 548, 599, 648, 691, 776, 861])

Real_T = np.size(Real_S)

UCI_Inicial = 5300
UCI_Fase_1 = UCI_Inicial
UCI_Fase_2 = UCI_Inicial + 2500
UCI_Fase_3 = UCI_Fase_2 + 2500
UCI_Fase_4 = UCI_Fase_3 + 2176

beta = 0.0878610407
alpha = 0.0474115920
gamma_1 = 0.9238
gamma_2 = 0.0509
gamma_3 = 0.0253
kappa = 0.0467383215

# Number of Individuals in each Set:
#    S is the Number of suceptible individuals at time t
#    C is the Number of carrier individuals at time t
#    IC is the Number of infected individuals staying at home at time t
#    IH is the Number of infected individuals staying at a hospital at time t
#    IU is the Number of infected individuals in an ICU at time t
#    R is the Number of removed/recovered individuals at time t
S = N - 2; C = 2; IC = 0; IH = 0; IU = 0; R = 0

# Lower case indicates the same set but as a rate over the total population
s = S/N; c = C/N; ic = IC/N; ih = IH/N; iu = IU/N; r = R/N

def covid_19(x, t):
    return -beta * x[0] * x[1],                                         \
            beta * x[0] * x[1] - alpha * x[1]

# Set up the figure the way we want it to look
plt.figure()
plt.clf()
plt.title('Covid - 19')

# Outer trajectories
phase_plot(
    covid_19,
    X0=[[S, C], ],
    T=np.linspace(0, 1000, 10000), #specigy simulation length and time resolution
    logtime=(0, .7) # plot arrows specity nulber of arrows per trajectory and distance between arrows
)

plt.show()
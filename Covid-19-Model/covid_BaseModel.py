import numpy as np
import matplotlib.pyplot as plt
from control import *

N = 49648685

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

beta = 0.0878610407
alpha = 0.0474115920
gamma_1 = 0.9238
gamma_2 = 0.0509
gamma_3 = 0.0253
kappa = 0.0467383215

S = N - 2; C = 2; IC = 0; IH = 0; IU = 0; R = 0
s = S/N; c = C/N; ic = IC/N; ih = IH/N; iu = IU/N; r = R/N

def covid_19(x):
    return -beta * x[0] * x[1],                                         \
            beta * x[0] * x[1] - alpha * x[1],                          \
            gamma_1 * (alpha * x[1] - kappa * x[2]),                    \
            gamma_2 * (alpha * x[1] - kappa * x[3]),                    \
            gamma_3 * (alpha * x[1] - kappa * x[4]),                    \
            kappa * (gamma_1 * x[2] + gamma_2 * x[3] + gamma_3 * x[4])

T = 1000

t = np.linspace(0, T, T + 1)

X = np.zeros((T + 1, 6))

X[0, :] = [s, c, ic, ih, iu, r]

for i in range(T):
        dX = covid_19(X[i,:])
        X[i + 1,:] = X[i] + dX

OA = 4

fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
fig.suptitle("Comparison Simulation - Real Data", fontsize=28)
axs[1].set_xlabel("Time (days)", fontsize=22)
axs[1].set_ylabel("Amount of People", fontsize=22)
axs[1].plot(t[0:Real_T*OA]/OA, X[0:Real_T*OA, 1]*N)
axs[1].plot(t[0:Real_T*OA]/OA, X[0:Real_T*OA, 2]*N)
axs[1].plot(t[0:Real_T*OA]/OA, X[0:Real_T*OA, 3]*N)
axs[1].plot(t[0:Real_T*OA]/OA, X[0:Real_T*OA, 4]*N)
axs[1].plot(t[0:Real_T*OA]/OA, X[0:Real_T*OA, 5]*N)
axs[1].legend(["Carriers", "Low-Risk Infected", "Moderate-Risk Infected", "High-Risk Infected", "Removed"], fontsize=18)
axs[0].set_xlabel("Time (days)", fontsize=22)
axs[0].set_ylabel("Amount of People", fontsize=22)
axs[0].plot(t[0:Real_T], Real_C)
axs[0].plot(t[0:Real_T], Real_I_C)
axs[0].plot(t[0:Real_T], Real_I_H)
axs[0].plot(t[0:Real_T], Real_I_U)
axs[0].plot(t[0:Real_T], Real_R)
axs[0].legend(["Real Carriers", "Real Low-Risk Infected", "Real Moderate-Risk Infected", "Real High-Risk Infected", "Real Removed"], fontsize=18)
plt.show()

fig = plt.figure()
plt.title("Simulation over the Span of 250 days", fontsize=28)
plt.xlabel("Time (days)", fontsize=22)
plt.ylabel("Amount of People", fontsize=22)
plt.plot(t/OA, X[:,0]*N)
plt.plot(t/OA, X[:,1]*N)
plt.plot(t/OA, X[:,2]*N)
plt.plot(t/OA, X[:,3]*N)
plt.plot(t/OA, X[:,4]*N)
plt.plot(t/OA, X[:,5]*N)
plt.legend(["Susceptible", "Carriers", "Low-Risk Infected", "Moderate-Risk Infected", "High-Risk Infected", "Removed"], fontsize=18)
plt.show()

UCI_Fase_1 = 5300
UCI_Fase_2 = UCI_Fase_1 + 2500
UCI_Fase_3 = UCI_Fase_2 + 2500
UCI_Fase_4 = UCI_Fase_3 + 2176

plt.plot(t/OA, X[:,4]*N)
plt.title("Comparison Simulation - Health Care Capacity", fontsize=28)
plt.xlabel("Time (days)", fontsize=22)
plt.ylabel("Amount of People (Log)", fontsize=22)
plt.hlines(UCI_Fase_1, 0, T/OA,  linestyle='dashed', color='k')
plt.hlines(UCI_Fase_2, 0, T/OA,  linestyle='dashed', color='r')
plt.hlines(UCI_Fase_3, 0, T/OA,  linestyle='dashed', color='g')
plt.hlines(UCI_Fase_4, 0, T/OA,  linestyle='dashed', color='m')
plt.legend(["High-Risk Infected", "Phase 1 UCI Capacity", "Phase 2 UCI Capacity", "Phase 3 UCI Capacity", "Phase 4 UCI Capacity"], fontsize=18, loc="lower left")
plt.show()

print("Min S:", np.min(X[:,0])*N)
print("Max C:", np.max(X[:,1])*N)
print("Max Ic:", np.max(X[:,2])*N)
print("Max Ih:", np.max(X[:,3])*N)
print("Max Iu:", np.max(X[:,4])*N)
print("Max R:", np.max(X[:,5])*N)

f1, = np.where(X[:,4]*N > UCI_Fase_1)
f2, = np.where(X[:,4]*N > UCI_Fase_2)
f3, = np.where(X[:,4]*N > UCI_Fase_3)
f4, = np.where(X[:,4]*N > UCI_Fase_4)

print("Supera Fase 1:", min(f1)/OA)
print("Supera Fase 2:", min(f2)/OA)
print("Supera Fase 3:", min(f3)/OA)
print("Supera Fase 4:", min(f4)/OA)

print(np.min(X[:,0]))
s_e = np.min(X[:,0])

A = np.array([[0,      -beta*s_e,              0,             0,              0, 0],
              [0, beta*s_e-alpha,              0,             0,              0, 0],
              [0,  gamma_1*alpha, -kappa*gamma_1,             0,              0, 0],
              [0,  gamma_2*alpha,              0,-kappa*gamma_2,              0, 0],
              [0,  gamma_3*alpha,              0,             0, -kappa*gamma_3, 0],
              [0,              0,  kappa*gamma_1, kappa*gamma_2,  kappa*gamma_3, 0]])
B = np.array([[-1], [1], [0], [0], [0], [0]])
C = np.array([0, 0, 0, 0, 1, 0])

# Check Observability of the system
Wo = obsv(A, C)
print("Wo = ", Wo)
print("Det(Wo) = ", np.linalg.det(Wo))
# the system is not observable

# Check controllability/reachability of the system
Wc = ctrb(A, B)
print("Wc = ", Wc)
print("Det(Wc) = ", np.linalg.det(Wc))
# the system is controllable
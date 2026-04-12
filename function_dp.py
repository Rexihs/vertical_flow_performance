import numpy as np
import math

def vniigaz_vertical_one(lamda_tube, diameter,
               velocity_liquid, velocity_gas,
               density_liquid, density_gas,
               pressure, sigma,
               q_liquid, q_gas):
    """
    Функция для расчёта течения ГЖП для вертикальных скважин
    по методике ВНИИГАЗ (модель 1)
    """

    # Вспомогательные параметры модели
    Afr = 319.3 * diameter - 16.666
    Bfr = 0.06134 / diameter

    # Формула из призентации, для угла наклона скважины 10 > alfa
    # Cfr = (2.2*10**-7 / diameter**2) + (1.364*10**3 / diameter)

    # Формула из Excel для угла наклона скважины 95 > alfa > 10
    Cfr = 0.0022 / (diameter * 100) ** 2 + 0.1364 / diameter / 100

    Dfr = (1.12 * 10**-3 / diameter**2) + (4.127 * 10**-2 / diameter) + 0.9956
    Efr = (5.213 * 10**-3 / diameter) + 0.0659

    af = 0.119445 / (diameter * 100) ** (8 / 3) + 0.000236
    bf = 0.3553 / diameter / diameter / 10000 + 0.0014
    i0 = af * (q_liquid*3600000) ** (2 / 3) + bf # Расход жидкости в л/ч

    # Расчёт параметров фруда
    Fr0 = Afr * i0 + Bfr
    Fr1 = Fr0 - Efr
    Fr = density_gas * velocity_gas ** 2 / density_liquid / 9.80665 / diameter

    i1 = i0 + lamda_tube / 2 * (Fr - Fr0)
    i2 = i0

    if Fr > Fr0:
        i = i1
    elif Fr < Fr1:
        Gfr = Cfr * ((Fr1 - Fr) / (Fr + 0.001)) ** Dfr + 1.5 * i0 / (1 - i0)
        i = Gfr / (Gfr + 1.5)
    else:
        i = i2

    dp_dz = (i * density_liquid + density_gas) * 9.80665
    return dp_dz

def vniigaz_vertical_two(lamda_tube, diameter,
                     velocity_liquid, velocity_gas,
                     density_liquid, density_gas,
                     pressure, sigma,
                     q_liquid, q_gas):
    """
    Функция для расчёта течения ГЖП по методике ВНИИГАЗ (модель 2) 
    для вертикальных скважин.
    """

    Frliq = velocity_liquid**2 / 9.80665 / diameter
    Eo = density_liquid * 9.80665 * diameter**2 / sigma
    D0 = diameter / 0.055
    Bu = Frliq**(1/3) * Eo**(2/3) / D0**2
    We = density_liquid * diameter * velocity_liquid**2 / sigma

    Fr0 = 1.15 - 1.15 * (1 - 1/D0) * np.exp(-4.6 * We**0.5)
    Fr1 = Fr0 - 0.0948/D0 - 0.0659
    Cliq = 0.0248/D0 + 0.0000727/D0**2
    Dliq = 0.996 + 0.75/D0 + 0.397/D0**2
    i0 = lamda_tube/2 * Fr0 + 0.00667 * Bu - 0.0012

    Fr = density_gas * velocity_gas**2 / density_liquid / 9.80665 / diameter
    
    if Fr > Fr0:
        i = i0 + lamda_tube/2 * (Fr - Fr0)
    elif Fr < Fr1:
        Gfr = Cliq * ((Fr1 - Fr) / (Fr + 0.001))**Dliq + 1.5 * i0 / (1 - i0)
        i = Gfr / (Gfr + 1.5)
    else:
        i = i0

    dp_dz = (i * density_liquid + density_gas) * 9.80665
    return dp_dz

def vniigaz_inclined(lamda_tube, diameter, alfa,
                    velocity_liquid, velocity_gas,
                    density_liquid, density_gas,
                    pressure, sigma,
                    q_liquid, q_gas):
    """
    Функция для расчёта течения ГЖП для наклонных скважин
    по методике ВНИИГАЗ.
    """

    # Вспомогательные параметры модели
    Afr = 3.1934 * diameter * 100 - 16.666
    BFr = 6.1343 / diameter / 100
    CFr = 0.0022 / (diameter * 100) ** 2 + 0.1364 / diameter / 100
    DFr = 12 / diameter / diameter / 10000 + 4.1273 / diameter / 100 + 0.9956
    Efr = 0.5213 / diameter / 100 + 0.0659

    # i0 из базовой формулы
    af = 0.119445 / (diameter * 100) ** (8 / 3) + 0.000236
    bf = 0.3553 / diameter / diameter / 10000 + 0.0014
    i0 = af * (q_liquid*3600000) ** (2 / 3) + bf # Расход жидкости в л/ч

    Fr0 = Afr * i0 + BFr
    Fr1 = Fr0 - Efr

    cos_alfa = math.cos(math.pi * alfa / 180.0)
    i0_alfa = i0 * (cos_alfa) ** 0.6
    
    Fr0_alfa = Fr0 * (90.0 - alfa) / 73.0 + 0.8 * np.sin(math.pi * (alfa - 17) / 73)
    Fr1_alfa = Fr0_alfa - Efr

    Fr = density_gas * velocity_gas ** 2 / density_liquid / 9.80665 / diameter

    if alfa > 17:
        # Для углов > 17° используется повернутая коррекция
        if Fr > Fr0_alfa:
            i = i0_alfa + lamda_tube / 2 * (Fr - Fr0_alfa)
        elif Fr < Fr1_alfa:
            Gfr = CFr * ((Fr1_alfa - Fr) / (Fr + 0.001)) ** DFr + 1.5 * i0_alfa / (1 - i0_alfa)
            i = Gfr / (Gfr + 1.5)
        else:
            i = i0_alfa
    else:
        # Для углов <= 17° обычная вертикальная модель
        if Fr > Fr0:
            i = i0 + lamda_tube / 2 * (Fr - Fr0)
        elif Fr < Fr1:
            Gfr = CFr * ((Fr1 - Fr) / (Fr + 0.001)) ** DFr + 1.5 * i0 / (1 - i0)
            i = Gfr / (Gfr + 1.5)
        else:
            i = i0
    print(i)
    # Финальный градиент давления с учётом наклона
    dp_dz = (i * density_liquid + density_gas * cos_alfa) * 9.80665
    return dp_dz

# --!! Внимание полностью проверить расчёт горизонталного потока !!--
# --!! Внимание полностью проверить расчёт горизонталного потока !!--
def vniigaz_horizontal(lamda_tube, diameter, alfa,
                            velocity_liquid, velocity_gas,
                            density_liquid, density_gas,
                            pressure, sigma,
                            q_liquid, q_gas):
    """
    Функция для расчёта течения ГЖП по методике ВНИИГАЗ для горизонтальных скважин.
    """
    if alfa >= 90:
        print('Error, alfa >= 90')
        return None

    gamma = 90 - alfa
    a = 0.0928 * gamma**(2/3) * (q_liquid * 3600000)**0.0913
    b = 8.64 / (q_liquid * 3600000)**0.245
    Fr0 = a * np.exp(-b * diameter)
    di = 0.185 * (1 - np.exp(0.15 * alfa - 13.5)) * diameter
    i1 = lamda_tube / 2 * Fr0 + di
    
    # Исправлено: используем math.cos для радиан, а не np.cos для градусов
    cos_alfa_rad = math.cos(alfa * math.pi / 180)
    C = -math.log(i1 / cos_alfa_rad) / Fr0**(1/3)

    Fr = density_gas * velocity_gas**2 / density_liquid / 9.80665 / diameter
    
    if Fr > Fr0:
        # iright
        i = lamda_tube / 2 * Fr
    else:
        # ileft
        i = cos_alfa_rad * np.exp(-C * Fr**(1/3))

    dp_dz = (i * density_liquid + density_gas * cos_alfa_rad) * 9.80665
    return dp_dz
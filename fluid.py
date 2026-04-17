import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

class Fluid:
    """Класс для расчёта свойств природного газа по методике GERG-91 мод."""

    Pstd = 0.101325  # стандартное давление, МПа
    Tstd = 293.15  # стандартная температура, К

    def __init__(self, rho_c: float, xa: float, xy: float, pvt_data: pd.DataFrame):
        """
        Параметры
        ----------
        rho_c : float
            Плотность газа в стандартных условиях, кг/м³.
        xa : float
            Мольная доля азота (N₂).
        xy : float
            Мольная доля диоксида углерода (CO₂).
        pvt_data : pd.DataFrame
            PVT-данные в формате pd.DataFrame с колонками 'pressure' бар, 'fvs' дол.ед., 'viscosity' мПа·с.
        """

        self.rho_c = rho_c # кг/м3 в ст. усл
        self.xa = xa/100 # дол.ед
        self.xy = xy/100 # дол.ед
        self.pvt_data = pvt_data

    def get_Z(self, P: float, T: float) -> float:
        """
        Рассчитать коэффициент сверхсжимаемости Z по методике GERG-91 мод.

        Параметры
        ----------
        P : float
            Давление, МПа.
        T : float
            Температура, K.

        Возвращает
        ----------
        float
            Коэффициент сверхсжимаемости Z.
        """

        # Блок проверки на условия применимости
        if not(self.rho_c > 0.668 and self.rho_c < 0.700):
            print(f'Attention - плотность газа в не диапазона 0.668 < ro < 0.700 кг/м3, ro:{self.rho_c}')
        if not(T > 250 and T < 330):
            print(f'Attention - температура газа в не диапазона 250 < T < 340 K, K:{T}')
        if not(P > 0 and P < 12):
            if P > 0 and P < 30:
                print(f'Attention - давление газа в не диапазона 0 < P < 12 МПа, P:{P}, погрешность расчёта +/- 3,0%')
            else:
                print(f'Attention - давление газа в не диапазона 0 < P < 30 МПа, P:{P}')

        xe = 1 - self.xa - self.xy # дол.ед, эквивалент углеводорода
        zc = 1-(0.0741*self.rho_c-0.006-0.063*self.xa-0.0575*self.xy)**2 # дол.ед, Фактор сжимаемости при стандартных условиях
        Me = (24.05525*zc*self.rho_c-28.0135*(self.xa)-44.01*(self.xy))/xe # г/моль, Молярная масса эквивалетного углеводорода

        H = 128.64 + 47.479 * Me
        Cx = 0.92 + 0.0013 * (T - 270)
        Bx = 0.72 + 1.875*10**-5 * (320 - T)**2

        C233 = 3.58783*10**-3 + 8.06674*10**-6 * T - 3.25798*10**-8 * T**2
        C223 = 5.52066*10**-3 - 1.68609*10**-5 * T + 1.57169*10**-8 * T**2

        C3 = 2.0513*10**-3 + 3.4888*10**-5*T - 8.3703*10**-8 * T**2
        C2 = 7.8498*10**-3 - 3.9895*10**-5*T + 6.1187*10**-8 * T**2
        C1 = (-0.302488 + 1.95861*10**-3*T - 3.16302*10**-6 * T**2
              + (6.46422*10**-4 - 4.22876*10**-6*T + 6.88157*10**-9*T**2) * H
              + (-3.32805*10**-7 + 2.2316*10**-9*T - 3.67713*10**-12*T**2) * H**2)

        B3 = -0.86834 + 4.0376 * 10**-3 * T - 5.1657 * 10**-6 * T**2
        B23 = -0.339693 + 1.61176 * 10**-3 * T - 2.04429 * 10**-6 * T**2
        B2 = -0.1446 + 7.4091 * 10**-4 * T - 9.1195 * 10**-7 * T**2
        B1 = (-0.425468 + 2.865e-3 * T - 4.62073e-6 * T ** 2
               + (8.77118e-4 - 5.56281e-6 * T + 8.8151e-9 * T ** 2) * H
               + (-8.24747e-7 + 4.31436e-9 * T - 6.08319e-12 * T ** 2) * H ** 2)

        Cm = (xe**3 * C1 + 3 * xe**2 * self.xa * Cx * (C1**2 * C2)**(1/3) + 2.76 * xe**2 * self.xy * (C1**2 * C3)**(1/3)
              + 3 *xe*self.xa**2*Cx*(C1*C2**2)**(1/3) + 6.6*xe*self.xa*self.xy*(C1 * C2 * C3)**(1/3) + 2.76 * xe * self.xy**2 * (C1 * C3**2)**(1/3)
              + self.xa**3* C2 + 3 *self.xa**2 * self.xy*C223 + 3 * self.xa*self.xy**2*C233 + self.xy**3 * C3)

        Bm = (xe**2*B1 + xe*self.xa*Bx*(B1+B2)-1.73*xe*self.xy*(B1*B3)**0.5
              +self.xa**2*B2+2*self.xa*self.xy*B23+self.xy**2*B3)

        # GERG-91 mod, P в МПа
        b = 10**3 * (P / (2.7715*T))
        C0 = b**2 * Cm
        B0 = b * Bm
        A1 = 1 + B0
        A0 = 1 + 1.5 * (B0 + C0)
        A2 = (A0 - (A0**2 - A1**3)**0.5)**(1/3)

        Z = (1 + A2 + A1/A2) / 3

        if np.iscomplex(Z):
            print(f'Внимание! Z принимает недействительные заничения при P:{P} МПа T:{T} К, return 1')
            return 1

        return Z

    def get_Bg(self, P: float, T: float) -> float:
        """
        Рассчитать объёмный коэффициент расширения газа Bg.

        Bg = (Pstd * Z * T) / (P * Tstd)

        Параметры
        ----------
        P : float
            Давление, МПа.
        T : float
            Температура, K.

        Возвращает
        ----------
        float
            Объёмный коэффициент расширения Bg.
        """

        Bg = (self.Pstd * self.get_Z(P, T) * T) / ((P) * self.Tstd)

        return Bg
    
    def get_ro(self, P: float, T,) -> float:      # плотность [кг/м³]
        """
        Расчёт плотости газа ro.
        ----------
        ro = rostd*P*/T/z*Tstd*1000000/Pstd

        Параметры
        ----------
        P : float
            Давление, МПа.
        T : float
            Температура, МПа

        Возвращает
        ----------
        float
            Плотность газа, кг/м³.
        """

        # ro = self.rho_c * (P / self.Pstd) * (self.Tstd / T) / self.get_Z(P, T)
        ro = self.rho_c / self.get_fvf(P)
        return ro

    def get_fvf(self, pressure: float) -> float:
        """Получить значение FVF (Formation Volume Factor) (дол.ед.) при заданном давлении (МПа)."""
        return np.interp(pressure*10, self.pvt_data['pressure'], self.pvt_data['fvf'])

    def get_viscosity(self, pressure: float) -> float:
        """Получить значение вязкости (Па·с) при заданном давлении (МПа)."""
        return (np.interp(pressure*10, self.pvt_data['pressure'], self.pvt_data['viscosity']))*10**-3
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from fluid import Fluid
from function_dp import lyamda, vniigaz_horizontal, vniigaz_inclined


class MainWindow(QMainWindow):
    DEFAULT_SIGMA = 0.072
    DEFAULT_GAS_DENSITY = 0.67
    DEFAULT_XA = 0.8858
    DEFAULT_XY = 0.0668

    def __init__(self):
        super().__init__()

        self.setWindowTitle("VFP Generator")
        self.setGeometry(100, 100, 1100, 850)

        self.well_data = None
        self.pvt_data = None
        self.gdi_data = None
        self.GDI_data = None
        self.vfp_data = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        file_buttons_layout = QHBoxLayout()
        btn_load_well = QPushButton("Загрузить инклинометрию")
        btn_load_well.clicked.connect(self.load_well)
        btn_load_pvt = QPushButton("Загрузить PVT")
        btn_load_pvt.clicked.connect(self.load_pvt)
        btn_load_gdi = QPushButton("Загрузить GDI")
        btn_load_gdi.clicked.connect(self.load_gdi)

        file_buttons_layout.addWidget(btn_load_well)
        file_buttons_layout.addWidget(btn_load_pvt)
        file_buttons_layout.addWidget(btn_load_gdi)
        layout.addLayout(file_buttons_layout)

        self.q_input = QLineEdit("50,100,150,250")
        self.thp_input = QLineEdit("5,10,20")
        self.wgr_input = QLineEdit("0,1e-5,1e-4")
        self.d_input = QLineEdit("0.1")
        self.delta_input = QLineEdit("0.0078")
        self.density_input = QLineEdit("1000")
        self.temp_thp_input = QLineEdit("293")
        self.temp_bhp_input = QLineEdit("310")

        params_layout = QGridLayout()
        params_layout.setHorizontalSpacing(12)
        params_layout.setVerticalSpacing(8)
        self._add_compact_field(params_layout, 0, 0, "Значения дебитов:", self.q_input)
        self._add_compact_field(params_layout, 0, 2, "Значение Руcт:", self.thp_input)
        self._add_compact_field(params_layout, 0, 4, "Значение WGR:", self.wgr_input)
        self._add_compact_field(params_layout, 0, 6, "Внутренний диаметр:", self.d_input)
        self._add_compact_field(params_layout, 1, 0, "Шероховатость:", self.delta_input)
        self._add_compact_field(params_layout, 1, 2, "Плотность жидк.:", self.density_input)
        self._add_compact_field(params_layout, 1, 4, "Устьевая T:", self.temp_thp_input)
        self._add_compact_field(params_layout, 1, 6, "Забойная T:", self.temp_bhp_input)
        for column in range(8):
            params_layout.setColumnStretch(column, 1 if column % 2 == 0 else 2)
        layout.addLayout(params_layout)

        action_buttons_layout = QHBoxLayout()
        btn_vfp = QPushButton("Рассчитать VFP")
        btn_vfp.clicked.connect(self.calculate_vfp)
        btn_gdi = QPushButton("Рассчитать GDI")
        btn_gdi.clicked.connect(self.calculate_gdi)
        btn_adapt = QPushButton("Адаптировать delta0 и density_liquid")
        btn_adapt.clicked.connect(self.optimize_gdi_parameters)

        action_buttons_layout.addWidget(btn_vfp)
        action_buttons_layout.addWidget(btn_gdi)
        action_buttons_layout.addWidget(btn_adapt)
        layout.addLayout(action_buttons_layout)

        info_layout = QHBoxLayout()
        self.optimization_result_label = QLabel("Оптимальные параметры: ещё не рассчитаны")
        info_layout.addWidget(self.optimization_result_label, stretch=3)
        info_layout.addWidget(QLabel("WGR для графиков:"), stretch=0)
        self.wgr_selector = QComboBox()
        self.wgr_selector.currentIndexChanged.connect(self.update_vfp_plot)
        info_layout.addWidget(self.wgr_selector, stretch=1)
        layout.addLayout(info_layout)

        self.table = QTableWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.table)

        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.addWidget(scroll)
        content_splitter.addWidget(self.canvas)
        content_splitter.setStretchFactor(0, 4)
        content_splitter.setStretchFactor(1, 6)
        layout.addWidget(content_splitter, stretch=1)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(10)
        self.slider.valueChanged.connect(self.update_plot)
        layout.addWidget(self.slider)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _add_compact_field(self, grid, row, column, label_text, widget):
        grid.addWidget(QLabel(label_text), row, column)
        grid.addWidget(widget, row, column + 1)

    # =========================
    # ЗАГРУЗКА ФАЙЛОВ
    # =========================
    def load_txt_file(self, path):
        file_path = Path(path)
        if not file_path.exists():
            raise ValueError(f"Файл не найден: {path}")

        header = self._detect_header(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as file_obj:
                df = pd.read_csv(
                    file_obj,
                    sep=r"\s+",
                    engine="python",
                    comment="#",
                    on_bad_lines="skip",
                    header=header,
                    skip_blank_lines=True,
                )
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="cp1251") as file_obj:
                df = pd.read_csv(
                    file_obj,
                    sep=r"\s+",
                    engine="python",
                    comment="#",
                    on_bad_lines="skip",
                    header=header,
                    skip_blank_lines=True,
                )
        except Exception as exc:
            raise ValueError(f"Не удалось прочитать файл: {exc}") from exc

        if df.empty:
            raise ValueError("Файл прочитан, но DataFrame пустой.")

        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if df.empty:
            raise ValueError("После очистки файл не содержит полезных данных.")

        if header is None:
            df.columns = self._make_default_columns(df.shape[1])
        else:
            df.columns = self._make_unique_columns(df.columns)

        df.columns = [self._normalize_column_name(col) for col in df.columns]
        df = self._drop_separator_rows(df)
        df = self._convert_numeric_like_columns(df)
        df = df.reset_index(drop=True)

        if df.empty:
            raise ValueError("После обработки файл не содержит ни одной полезной строки.")

        return df

    def _detect_header(self, file_path):
        try_encodings = ("utf-8", "cp1251")
        for encoding in try_encodings:
            try:
                with open(file_path, "r", encoding=encoding) as file_obj:
                    for line in file_obj:
                        stripped = line.strip()
                        if not stripped or stripped.startswith("#"):
                            continue
                        tokens = re.split(r"\s+", stripped)
                        alpha_tokens = [token for token in tokens if re.search(r"[A-Za-zА-Яа-я]", token)]
                        return 0 if alpha_tokens else None
            except UnicodeDecodeError:
                continue
        return 0

    def _make_default_columns(self, count):
        columns = [f"COL_{index + 1}" for index in range(count)]
        defaults = ["MD", "TVD", "INCL"]
        for index, name in enumerate(defaults):
            if index < count:
                columns[index] = name
        return columns

    def _make_unique_columns(self, columns):
        result = []
        seen = {}
        for column in columns:
            normalized = self._normalize_column_name(column) or "COL"
            suffix = seen.get(normalized, 0)
            result.append(normalized if suffix == 0 else f"{normalized}_{suffix + 1}")
            seen[normalized] = suffix + 1
        return result

    def _normalize_column_name(self, name):
        text = str(name).strip()
        text = re.sub(r"[^0-9A-Za-zА-Яа-я_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        upper_text = text.upper()

        alias_map = {
            "MD": ["MD", "MEASUREDDEPTH", "DEPTH"],
            "TVD": ["TVD", "TRUEVERTICALDEPTH"],
            "INCL": ["INCL", "INC", "ANGLE", "DEVIATION"],
            "THP": ["THP", "WHP"],
            "FLO": ["FLO", "FLOW", "QGAS", "RATE", "DEBIT"],
            "WGR": ["WGR"],
            "BHP": ["BHP"],
            "DP": ["DP", "DELTAP", "PRESSUREDROP"],
            "PRESSURE": ["PRESSURE"],
            "FVF": ["FVF", "FVS", "BO"],
            "VISCOSITY": ["VISCOSITY", "MU", "VISC"],
        }

        for canonical, variants in alias_map.items():
            if any(variant in upper_text for variant in variants):
                return canonical

        return upper_text or "COL"

    def _drop_separator_rows(self, df):
        def is_separator_row(row):
            values = [str(value).strip() for value in row.tolist() if pd.notna(value)]
            if not values:
                return True
            return all(re.fullmatch(r"[-_=]+", value) for value in values)

        return df.loc[~df.apply(is_separator_row, axis=1)].reset_index(drop=True)

    def _convert_numeric_like_columns(self, df):
        for column in df.columns:
            series = df[column].astype(str).str.replace(",", ".", regex=False).str.strip()
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() > 0:
                df[column] = numeric
            else:
                df[column] = series
        return df

    def _prepare_well_dataframe(self, df):
        prepared = df.copy()

        if "TVD" not in prepared.columns and "COL_5" in prepared.columns:
            prepared = prepared.rename(columns={"COL_5": "TVD"})
        if "INCL" not in prepared.columns and "COL_9" in prepared.columns:
            prepared = prepared.rename(columns={"COL_9": "INCL"})

        required_columns = ["MD", "TVD", "INCL"]
        missing = [column for column in required_columns if column not in prepared.columns]
        if missing:
            raise ValueError(f"В файле инклинометрии не найдены обязательные колонки: {', '.join(missing)}")

        prepared = prepared[required_columns].dropna().reset_index(drop=True)
        if prepared.empty:
            raise ValueError("Инклинометрия загружена, но после очистки не осталось строк.")

        return prepared

    def _prepare_pvt_dataframe(self, df):
        prepared = df.copy()
        rename_map = {
            "PRESSURE": "pressure",
            "FVF": "fvf",
            "VISCOSITY": "viscosity",
        }
        prepared = prepared.rename(columns={key: value for key, value in rename_map.items() if key in prepared.columns})

        required_columns = ["pressure", "fvf", "viscosity"]
        missing = [column for column in required_columns if column not in prepared.columns]
        if missing:
            raise ValueError(f"В PVT файле не найдены обязательные колонки: {', '.join(missing)}")

        prepared = prepared[required_columns].dropna().reset_index(drop=True)
        if prepared.empty:
            raise ValueError("PVT файл загружен, но после очистки не осталось строк.")

        return prepared

    def _prepare_gdi_dataframe(self, df):
        prepared = df.copy()
        rename_map = {"DP": "dp"}
        prepared = prepared.rename(columns={key: value for key, value in rename_map.items() if key in prepared.columns})

        required_columns = ["THP", "FLO", "WGR", "BHP", "dp"]
        missing = [column for column in required_columns if column not in prepared.columns]
        if missing:
            raise ValueError(f"В GDI файле не найдены обязательные колонки: {', '.join(missing)}")

        prepared = prepared[required_columns].dropna().reset_index(drop=True)
        if prepared.empty:
            raise ValueError("GDI файл загружен, но после очистки не осталось строк.")

        return prepared

    def _select_file(self, title):
        return QFileDialog.getOpenFileName(
            self,
            title,
            "",
            "Data files (*.txt *.dev *.inc *.dat *.csv);;All files (*.*)",
        )[0]

    def _load_dataset(self, attr_name, title, success_label, prepare_func, legacy_attr=None):
        path = self._select_file(title)
        if not path:
            return

        try:
            df = self.load_txt_file(path)
            df = prepare_func(df)
            if df.empty:
                raise ValueError("Файл не содержит данных после загрузки.")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))
            return

        print(f"\n[{success_label}] {path}")
        print(df.head())

        setattr(self, attr_name, df)
        if legacy_attr:
            setattr(self, legacy_attr, df)

        QMessageBox.information(self, "Успех", "Файл успешно загружен")

    def load_well(self):
        self._load_dataset(
            "well_data",
            "Выберите файл инклинометрии",
            "well_data",
            self._prepare_well_dataframe,
        )

    def load_pvt(self):
        self._load_dataset(
            "pvt_data",
            "Выберите PVT файл",
            "pvt_data",
            self._prepare_pvt_dataframe,
        )

    def load_gdi(self):
        self._load_dataset(
            "gdi_data",
            "Выберите GDI файл",
            "gdi_data",
            self._prepare_gdi_dataframe,
            legacy_attr="GDI_data",
        )

    # =========================
    # ВСПОМОГАТЕЛЬНОЕ
    # =========================
    def parse_list(self, text):
        return [float(item.strip()) for item in text.split(",") if item.strip()]

    def create_fluid(self):
        if self.pvt_data is None:
            raise ValueError("Сначала загрузите PVT файл.")
        return Fluid(self.DEFAULT_GAS_DENSITY, self.DEFAULT_XA, self.DEFAULT_XY, self.pvt_data)

    def require_loaded_data(self, *names):
        missing = [name for name in names if getattr(self, name) is None]
        if missing:
            raise ValueError(f"Не загружены данные: {', '.join(missing)}")

    def get_input_parameters(self):
        return {
            "q_list": self.parse_list(self.q_input.text()),
            "thp_list": self.parse_list(self.thp_input.text()),
            "wgr_list": self.parse_list(self.wgr_input.text()),
            "diameter": float(self.d_input.text()),
            "delta0": float(self.delta_input.text()),
            "density_liquid": float(self.density_input.text()),
            "temp_thp": float(self.temp_thp_input.text()),
            "temp_bhp": float(self.temp_bhp_input.text()),
            "sigma": self.DEFAULT_SIGMA,
        }

    # =========================
    # РАСЧЁТЫ
    # =========================

    def calc_THP_from_BHP(self, well_data: pd, fluid: Fluid, BHP, Temp_THP, Temp_BHP,
                          Qgas, WGR, diameter, delta0, sigma, density_liquid):
    
        well_df = pd.DataFrame({
            'MD': well_data['MD'],
            'TVD': well_data['TVD'],
            'alfa': well_data['INCL'],
            'D': diameter,
            'Dp_dz': np.nan,
            'P': np.nan
        })

        # стартуем с забоя
        well_df.loc[len(well_df)-1, 'P'] = BHP

        # идём СНИЗУ ВВЕРХ
        for idx in well_df.index[::-1]:

            if idx == 0:
                break

            temp_idx = Temp_THP + (Temp_BHP - Temp_THP) / well_df['TVD'].iloc[-1] * well_df.loc[idx, 'TVD']

            P_local = well_df.loc[idx, 'P'] #МПа

            density_gas_idx = fluid.get_ro(P_local, temp_idx)

            velocity_liquid_idx = ((Qgas * 1000 / 86400) * WGR) / (np.pi * diameter**2 / 4)
            velocity_gas_idx = (Qgas * 1000 / 86400) * fluid.get_fvf(P_local) / (np.pi * diameter**2 / 4)

            lyamda_idx = lyamda(
                delta0=delta0,
                diameter=diameter,
                velocity_liquid=velocity_liquid_idx,
                velocity_gas=velocity_gas_idx,
                density_liquid=density_liquid,
                density_gas=density_gas_idx,
                sigma=sigma,
                viscosity_gas=fluid.get_viscosity(P_local)
            )

            if well_df.loc[idx, 'alfa'] > 84:
                dp_dz = vniigaz_horizontal(
                    lyamda_idx, diameter, well_df.loc[idx, 'alfa'],
                    velocity_liquid_idx, velocity_gas_idx,
                    density_liquid, density_gas_idx,
                    P_local, sigma, (Qgas * 1000 / 86400) * WGR
                )
            else:
                dp_dz = vniigaz_inclined(
                    lyamda_idx, diameter, well_df.loc[idx, 'alfa'],
                    velocity_liquid_idx, velocity_gas_idx,
                    density_liquid, density_gas_idx,
                    P_local, sigma, (Qgas * 1000 / 86400) * WGR
                )

            dL = well_df.loc[idx, 'MD'] - well_df.loc[idx-1, 'MD']

            # движение вверх → давление падает
            well_df.loc[idx-1, 'P'] = P_local - dp_dz * 1e-6 * dL

        return well_df.loc[0, 'P']  # THP
    
    # Правильная формула расчёта BHP:
    def well_BHP(self, well_data: pd, fluid: Fluid, THP_target,
                 Temp_THP, Temp_BHP,
                 Qgas, WGR, diameter, delta0, sigma, density_liquid):

        # начальные границы (важно!)
        BHP_min = THP_target
        BHP_max = THP_target + 50  # запас (МПа)

        tol = 1e-3
        max_iter = 50

        for _ in range(max_iter):

            BHP_mid = 0.5 * (BHP_min + BHP_max)

            THP_calc = self.calc_THP_from_BHP(
                well_data, fluid, BHP_mid,
                Temp_THP, Temp_BHP,
                Qgas, WGR,
                diameter, delta0, sigma, density_liquid
            )

            error = THP_calc - THP_target

            if abs(error) < tol:
                return BHP_mid

            # бисекция
            if error > 0:
                BHP_max = BHP_mid
            else:
                BHP_min = BHP_mid

        print("Warning: BHP not converged")
        return BHP_mid


    def adapt_gdi(self, gdi_data, delta0, density_liquid):
        params = self.get_input_parameters()
        fluid = self.create_fluid()

        df_data = pd.DataFrame(
            {
                "THP": gdi_data["THP"] / 10,
                "FLO": gdi_data["FLO"] / 1000,
                "WGR": gdi_data["WGR"],
                "BHP": gdi_data["BHP"] / 10,
                "BHP_calc": np.nan,
                "dp_calc": np.nan,
                "d_bhp": np.nan,
            }
        )

        for idx in df_data.index:
            bhp_calc = self.well_BHP(
                self.well_data,
                fluid,
                df_data.loc[idx, "THP"],
                params["temp_thp"],
                params["temp_bhp"],
                df_data.loc[idx, "FLO"],
                df_data.loc[idx, "WGR"],
                params["diameter"],
                delta0,
                params["sigma"],
                density_liquid,
            )
            df_data.loc[idx, "BHP_calc"] = bhp_calc
            df_data.loc[idx, "dp_calc"] = bhp_calc - df_data.loc[idx, "THP"]
            df_data.loc[idx, "d_bhp"] = bhp_calc - df_data.loc[idx, "BHP"]

        return df_data

    def objective(self, params):
        delta0, density_liquid = params
        if delta0 <= 0 or density_liquid <= 0:
            return 1e12

        try:
            df = self.adapt_gdi(self.gdi_data, delta0, density_liquid)
        except Exception:
            return 1e12

        return float(np.mean((df["BHP"] - df["BHP_calc"]) ** 2))

    def generate_vfp_table(self, q_gas_list, wgr_list, thp_list, well_data, fluid, temp_thp, temp_bhp, diameter, delta0, sigma, density_liquid):
        rows = []

        for i_wgr, wgr in enumerate(wgr_list, start=1):
            for i_thp, thp in enumerate(thp_list, start=1):
                row = {
                    "THP_i": i_thp,
                    "WGR_i": i_wgr,
                    "THP": thp,
                    "WGR": wgr,
                    "GFR_i": 1,
                    "ALQ_i": 1,
                }

                for i_q, q in enumerate(q_gas_list, start=1):
                    bhp = self.well_BHP(
                        well_data,
                        fluid,
                        thp,
                        temp_thp,
                        temp_bhp,
                        q,
                        wgr,
                        diameter,
                        delta0,
                        sigma,
                        density_liquid,
                    )
                    row[f"BHP_q{i_q}"] = bhp

                rows.append(row)

        return pd.DataFrame(rows)

    # =========================
    # ДЕЙСТВИЯ GUI
    # =========================
    def calculate_gdi(self):
        try:
            self.require_loaded_data("well_data", "pvt_data", "gdi_data")
            params = self.get_input_parameters()
            df = self.adapt_gdi(self.gdi_data, params["delta0"], params["density_liquid"])
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка расчёта GDI", str(exc))
            return

        print("\n[gdi_result]")
        print(df.head())
        self.show_table(df)
        QMessageBox.information(self, "Успех", "GDI успешно рассчитан")

    def optimize_gdi_parameters(self):
        try:
            self.require_loaded_data("well_data", "pvt_data", "gdi_data")
            initial = [float(self.delta_input.text()), float(self.density_input.text())]
            result = minimize(
                self.objective,
                initial,
                method="L-BFGS-B",
                bounds=[(1e-14, 1.0), (1.0, 5000.0)],
            )
            if not result.success:
                raise ValueError(result.message)

            opt_delta0, opt_density = result.x
            self.delta_input.setText(f"{opt_delta0:.12g}")
            self.density_input.setText(f"{opt_density:.12g}")
            self.optimization_result_label.setText(
                "Оптимальные параметры: "
                f"delta0 = {opt_delta0:.12g}, density_liquid = {opt_density:.12g}"
            )
            QMessageBox.information(
                self,
                "Адаптация завершена",
                f"Оптимальные параметры:\n"
                f"delta0 = {opt_delta0:.12g}\n"
                f"density_liquid = {opt_density:.12g}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка адаптации", str(exc))

    def calculate_vfp(self):
        try:
            self.require_loaded_data("well_data", "pvt_data")
            params = self.get_input_parameters()
            fluid = self.create_fluid()

            self.vfp_data = self.generate_vfp_table(
                q_gas_list=params["q_list"],
                wgr_list=params["wgr_list"],
                thp_list=params["thp_list"],
                well_data=self.well_data,
                fluid=fluid,
                temp_thp=params["temp_thp"],
                temp_bhp=params["temp_bhp"],
                diameter=params["diameter"],
                delta0=params["delta0"],
                sigma=params["sigma"],
                density_liquid=params["density_liquid"],
            )
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка расчёта VFP", str(exc))
            return

        print("\n[vfp_result]")
        print(self.vfp_data.head())

        self.show_table(self.vfp_data)
        self.populate_wgr_selector(params["wgr_list"])
        self.update_vfp_plot()
        QMessageBox.information(self, "Успех", "VFP успешно рассчитан")

    def populate_wgr_selector(self, wgr_list):
        self.wgr_selector.blockSignals(True)
        self.wgr_selector.clear()
        for index, wgr in enumerate(wgr_list, start=1):
            self.wgr_selector.addItem(f"{index}: {wgr}", index)
        self.wgr_selector.blockSignals(False)

    def update_vfp_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if self.vfp_data is None or self.vfp_data.empty or self.wgr_selector.count() == 0:
            ax.set_title("График VFP появится после расчёта")
            self.canvas.draw()
            return

        params = self.get_input_parameters()
        wgr_index = self.wgr_selector.currentData()
        if wgr_index is None:
            wgr_index = 1

        df_slice = self.vfp_data[self.vfp_data["WGR_i"] == wgr_index].reset_index(drop=True)
        if df_slice.empty:
            ax.set_title("Нет данных для выбранного WGR")
            self.canvas.draw()
            return

        for i_q, q in enumerate(params["q_list"], start=1):
            column_name = f"BHP_q{i_q}"
            if column_name in df_slice.columns:
                ax.plot(df_slice["THP"], df_slice[column_name], marker="o", label=f"Q={q}")

        ax.set_xlabel("THP, МПа")
        ax.set_ylabel("BHP, МПа")
        ax.set_title(f"VFP Table (WGR = {params['wgr_list'][wgr_index - 1]})")
        ax.grid(True)
        ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()

    # =========================
    # ТАБЛИЦА
    # =========================
    def show_table(self, df):
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(column) for column in df.columns])

        for i in range(len(df)):
            for j in range(len(df.columns)):
                self.table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))

    # =========================
    # ДЕМО-СЛАЙДЕР
    # =========================
    def update_plot(self):
        value = self.slider.value()
        if self.vfp_data is not None and not self.vfp_data.empty:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x = np.linspace(0, 100, 50)
        y = x * (value + 1)
        ax.plot(x, y)
        ax.set_title(f"Slider = {value}")
        ax.grid(True)
        self.figure.tight_layout()
        self.canvas.draw()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())

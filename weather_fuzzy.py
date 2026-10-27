import os
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ----------------------------------------------------------------------------
# Настройки
# ----------------------------------------------------------------------------
API_KEY = os.getenv("OWM_API_KEY") or "your-api"
HISTORY_YEARS = 5          # сколько лет истории грузим
SHOW_PLOTS = True          # False -> только сохранять картинки в файлы
OUT_DIR = Path("lab3_output")
OUT_DIR.mkdir(exist_ok=True)

CITIES = {
    "Kazan":            {"ru": "Казань",          "lat": 55.7887, "lon": 49.1221},
    "Moscow":           {"ru": "Москва",          "lat": 55.7558, "lon": 37.6173},
    "Saint Petersburg": {"ru": "Санкт-Петербург", "lat": 59.9343, "lon": 30.3351},
}
SEASONS = ["winter", "spring", "summer", "autumn"]
SEASON_RU = {"winter": "зима", "spring": "весна", "summer": "лето", "autumn": "осень"}


# ----------------------------------------------------------------------------
# Этап 1. Текущие данные из OpenWeatherMap
# ----------------------------------------------------------------------------
def get_current_weather(city, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": f"{city},RU",
        "appid": api_key,
        "units": "metric",
        "lang": "ru",
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        weather_info = {
            "city": city,
            "timestamp": datetime.now(),
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
            "wind_gust": data["wind"].get("gust", 0),
            "description": data["weather"][0]["description"],
        }
        return weather_info
    except Exception as e:
        print(f"Ошибка получения данных для {city}: {e}")
        return None


# ----------------------------------------------------------------------------
# Этап 2. Исторические данные (Open-Meteo Archive API)
# ----------------------------------------------------------------------------
def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


def get_history(city, years=HISTORY_YEARS, force=False):
    """Среднесуточные температура, влажность, ветер. Кэшируется в CSV."""
    path = OUT_DIR / f"history_{city.replace(' ', '_')}.csv"
    if path.exists() and not force:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    end = date.today() - timedelta(days=5)      # архив отстаёт на пару дней
    start = date(end.year - years, 1, 1)        # полные календарные годы
    params = {
        "latitude": CITIES[city]["lat"],
        "longitude": CITIES[city]["lon"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "wind_speed_unit": "ms",
        "timezone": "Europe/Moscow",
    }
    r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                     params=params, timeout=120)
    r.raise_for_status()
    hourly = pd.DataFrame(r.json()["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly = hourly.set_index("time").rename(columns={
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_speed",
    })
    daily = hourly.resample("D").mean().dropna()
    daily.index.name = "date"
    daily.to_csv(path)
    return daily


# ----------------------------------------------------------------------------
# Этап 3. Климатические нормы по сезонам
# ----------------------------------------------------------------------------
def compute_seasonal_norms(history):
    frames = []
    for city, df in history.items():
        d = df.copy()
        d["city"] = city
        d["season"] = d.index.month.map(get_season)
        frames.append(d)
    all_df = pd.concat(frames)
    norms = all_df.groupby(["city", "season"]).agg(
        temp_mean=("temperature", "mean"), temp_std=("temperature", "std"),
        hum_mean=("humidity", "mean"), hum_std=("humidity", "std"),
        wind_mean=("wind_speed", "mean"), wind_std=("wind_speed", "std"),
    ).round(2)
    return norms


def window_temp_norm(df, today, half_window=7):
    """Норма температуры на «сегодня» по истории в окне ±7 дней.
    Нужна, чтобы переходные периоды (конец октября и т.п.) не искажали сравнение."""
    doy = np.asarray(df.index.dayofyear)
    diff = np.abs(doy - today.timetuple().tm_yday)
    diff = np.minimum(diff, 366 - diff)
    sub = df[diff <= half_window]["temperature"]
    return sub.mean(), sub.std()


# ----------------------------------------------------------------------------
# Этап 4. Числа -> лингвистические значения (учёт города и сезона)
# Таблица: (верхняя граница, терм). Последний терм - для всех значений выше.
# ----------------------------------------------------------------------------
LING_TEMP = {
    "Kazan": {
        "winter": [(-25, "экстремально холодно"), (-15, "холодно"), (-5, "умеренно холодно"),
                   (0, "близко к нулю"), (None, "тепло для зимы")],
        "spring": [(-10, "холодно для весны"), (0, "заморозки"), (8, "прохладно"),
                   (15, "умеренно"), (22, "тепло"), (None, "жарко для весны")],
        "summer": [(15, "холодно для лета"), (20, "прохладно"), (25, "комфортно"),
                   (30, "тепло"), (None, "жарко")],
        "autumn": [(-5, "ранние морозы"), (3, "холодно"), (10, "прохладно"),
                   (18, "умеренно"), (None, "тепло для осени")],
    },
    "Moscow": {
        "winter": [(-25, "экстремальный мороз"), (-15, "сильный мороз"), (-7, "холодно"),
                   (0, "умеренный мороз"), (None, "оттепель")],
        "spring": [(-5, "холодно для весны"), (3, "прохладно"), (12, "умеренно"),
                   (20, "тепло"), (None, "жарко для весны")],
        "summer": [(12, "холодно для лета"), (18, "прохладно"), (23, "комфортно"),
                   (28, "тепло"), (None, "жарко (возможна волна жары)")],
        "autumn": [(-3, "ранние морозы"), (4, "холодно"), (11, "прохладно"),
                   (18, "умеренно"), (None, "тепло для осени")],
    },
    "Saint Petersburg": {
        "winter": [(-15, "морозно (редкость)"), (-5, "холодно"), (0, "около нуля"),
                   (None, "оттепель")],
        "spring": [(-5, "холодно для весны"), (2, "прохладно"), (10, "умеренно"),
                   (18, "тепло"), (None, "жарко для весны")],
        "summer": [(10, "очень холодно"), (15, "прохладно"), (20, "умеренно"),
                   (25, "тепло"), (None, "жарко")],
        "autumn": [(0, "морозно"), (5, "холодно"), (10, "прохладно"),
                   (16, "умеренно"), (None, "тепло для осени")],
    },
}


def numeric_to_linguistic(value, var_name, city_name, season):
    if var_name == "temperature":
        for upper, label in LING_TEMP[city_name][season]:
            if upper is None or value < upper:
                return label
    elif var_name == "humidity":
        if value < 30: return "сухо"
        elif value < 60: return "нормальная"
        elif value < 85: return "повышенная"
        else: return "очень высокая"
    elif var_name == "wind":
        if value < 4: return "слабый"
        elif value < 10: return "умеренный"
        elif value < 15: return "сильный"
        else: return "очень сильный"
    return "нет данных"


# ----------------------------------------------------------------------------
# Этап 5. Нечёткая система Мамдани
# ----------------------------------------------------------------------------
T_MIN, T_MAX = -30, 40

# Термы температуры (одни и те же имена во всех сезонах, но разные границы).
# 3 числа -> треугольная функция, 4 -> трапециевидная.
SEASON_TEMP_TERMS = {
    "winter": {"freezing": [-30, -30, -25, -15], "cold": [-25, -15, -5], "cool": [-12, -5, 2],
               "comfortable": [-3, 2, 8], "warm": [5, 10, 18], "hot": [15, 22, 40, 40]},
    "spring": {"freezing": [-30, -30, -15, -5], "cold": [-12, -5, 3], "cool": [0, 6, 12],
               "comfortable": [8, 14, 20], "warm": [17, 22, 27], "hot": [24, 30, 40, 40]},
    "summer": {"freezing": [-30, -30, 5, 10], "cold": [5, 10, 15], "cool": [12, 16, 21],
               "comfortable": [18, 22, 26], "warm": [24, 28, 32], "hot": [29, 33, 40, 40]},
    "autumn": {"freezing": [-30, -30, -10, -2], "cold": [-8, 0, 6], "cool": [3, 8, 13],
               "comfortable": [10, 15, 20], "warm": [17, 22, 27], "hot": [25, 30, 40, 40]},
}

# Экспертные поправки: местные жители адаптированы к своему климату
TEMP_SHIFT = {
    ("Kazan", "winter"): -3,            # привыкли к более суровой зиме
    ("Saint Petersburg", "summer"): -2,  # лето прохладнее, «комфортно» начинается раньше
}

# Чувствительность к ветру и влажности по городам
CITY_PROFILE = {
    "Kazan":            {"wind_strong": 9, "hum_high": 70},
    "Moscow":           {"wind_strong": 9, "hum_high": 70},   # шквалистые ветры
    "Saint Petersburg": {"wind_strong": 7, "hum_high": 65},   # море, опасный ветер у залива
}

TERM_RU = {
    "temperature": {"freezing": "очень холодно", "cold": "холодно", "cool": "прохладно",
                    "comfortable": "комфортно", "warm": "тепло", "hot": "жарко"},
    "humidity": {"low": "сухо", "normal": "нормально", "high": "влажно"},
    "wind_speed": {"calm": "штиль", "moderate": "умеренный", "strong": "сильный"},
    "comfort": {"bad": "плохо", "acceptable": "приемлемо", "good": "хорошо"},
}

# Баллы для автоматической генерации базы правил (читаемая «таблица эксперта»)
TEMP_SCORE = {"freezing": 0, "cold": 0, "cool": 2, "comfortable": 4, "warm": 3, "hot": 1}
HUM_PENALTY = {"low": 0, "normal": 0, "high": 1}
WIND_PENALTY = {"calm": 0, "moderate": 1, "strong": 2}


def _shift(params, d):
    """Сдвиг только внутренних точек (границы универсума -30 и 40 не трогаем)."""
    return [p if p in (T_MIN, T_MAX) else p + d for p in params]


def _mf(universe, params):
    return fuzz.trapmf(universe, params) if len(params) == 4 else fuzz.trimf(universe, params)


def _level(score):
    return "good" if score >= 3 else ("acceptable" if score >= 1 else "bad")


class FuzzyWeatherSystem:
    """Нечёткая система: (температура, влажность, ветер) -> комфорт [0..100]."""

    def __init__(self, city, season):
        self.city = city
        self.season = season
        prof = CITY_PROFILE[city]

        # Антецеденты и консеквент
        self.temperature = ctrl.Antecedent(np.arange(T_MIN, T_MAX + 1, 1), "temperature")
        self.humidity = ctrl.Antecedent(np.arange(0, 101, 1), "humidity")
        self.wind = ctrl.Antecedent(np.arange(0, 31, 1), "wind_speed")
        self.comfort = ctrl.Consequent(np.arange(0, 101, 1), "comfort")

        # Температура: термы зависят от сезона (+ поправка для города)
        shift = TEMP_SHIFT.get((city, season), 0)
        for name, params in SEASON_TEMP_TERMS[season].items():
            self.temperature[name] = _mf(self.temperature.universe, _shift(params, shift))

        # Влажность
        h = prof["hum_high"]
        self.humidity["low"] = fuzz.trapmf(self.humidity.universe, [0, 0, 25, 40])
        self.humidity["normal"] = fuzz.trimf(self.humidity.universe, [30, 55, h + 10])
        self.humidity["high"] = fuzz.trapmf(self.humidity.universe, [h, h + 15, 100, 100])

        # Ветер
        w = prof["wind_strong"]
        self.wind["calm"] = fuzz.trapmf(self.wind.universe, [0, 0, 2, 5])
        self.wind["moderate"] = fuzz.trimf(self.wind.universe, [2, 6, w + 2])
        self.wind["strong"] = fuzz.trapmf(self.wind.universe, [w, w + 5, 30, 30])

        # Комфорт
        self.comfort["bad"] = fuzz.trapmf(self.comfort.universe, [0, 0, 20, 40])
        self.comfort["acceptable"] = fuzz.trimf(self.comfort.universe, [25, 50, 75])
        self.comfort["good"] = fuzz.trapmf(self.comfort.universe, [60, 80, 100, 100])

        self.rules = self._build_rules()
        self.system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.system)

    def _build_rules(self):
        rules = []
        # Базовая матрица правил: 6 (температура) x 3 (влажность) x 3 (ветер) = 54
        for t, t_score in TEMP_SCORE.items():
            for h, h_pen in HUM_PENALTY.items():
                for wd, w_pen in WIND_PENALTY.items():
                    hum_pen, wind_pen = h_pen, w_pen
                    if t == "hot":                  # в жару влажность хуже, а лёгкий ветер помогает
                        hum_pen = 2 if h == "high" else 0
                        wind_pen = 0 if wd == "moderate" else w_pen
                    score = t_score - hum_pen - wind_pen
                    rules.append(ctrl.Rule(
                        self.temperature[t] & self.humidity[h] & self.wind[wd],
                        self.comfort[_level(score)]))

        # Экспертные правила для конкретных городов
        if self.city == "Saint Petersburg":
            rules.append(ctrl.Rule(self.humidity["high"] & self.wind["strong"],
                                   self.comfort["bad"]))
        if self.city == "Moscow":
            rules.append(ctrl.Rule(self.temperature["hot"] & self.humidity["high"],
                                   self.comfort["bad"]))
        if self.city == "Kazan":
            rules.append(ctrl.Rule(self.temperature["cool"] & self.wind["strong"],
                                   self.comfort["bad"]))
        return rules

    @staticmethod
    def _clip(var, value):
        return float(np.clip(value, var.universe.min(), var.universe.max()))

    def memberships(self, temp, hum, wind):
        """Фаззификация: степени принадлежности ко всем термам."""
        result = {}
        for var, val in ((self.temperature, temp), (self.humidity, hum), (self.wind, wind)):
            v = self._clip(var, val)
            result[var.label] = {
                t: float(fuzz.interp_membership(var.universe, var[t].mf, v)) for t in var.terms
            }
        return result

    def evaluate(self, temp, hum, wind):
        """Полный вывод Мамдани. Возвращает (балл комфорта, терм комфорта)."""
        self.simulation.input["temperature"] = self._clip(self.temperature, temp)
        self.simulation.input["humidity"] = self._clip(self.humidity, hum)
        self.simulation.input["wind_speed"] = self._clip(self.wind, wind)
        self.simulation.compute()
        score = float(self.simulation.output["comfort"])
        degrees = {t: float(fuzz.interp_membership(self.comfort.universe, self.comfort[t].mf, score))
                   for t in self.comfort.terms}
        label = max(degrees, key=degrees.get)
        return score, label


def format_memberships(deg, var_name, top=2, eps=0.05):
    items = sorted(deg[var_name].items(), key=lambda kv: -kv[1])
    items = [(k, v) for k, v in items if v > eps][:top]
    return ", ".join(f'"{TERM_RU[var_name][k]}" ({v:.2f})' for k, v in items) or "-"


# ----------------------------------------------------------------------------
# Сравнение с нормой и аномалии
# ----------------------------------------------------------------------------
def anomaly_label(z):
    az = abs(z)
    if az < 1:
        return "в норме"
    side = "выше" if z > 0 else "ниже"
    return f"заметно {side} нормы" if az < 2 else f"аномально {side} нормы"


def detect_anomalies(w, norm_row, win_mean, win_std):
    out = {}
    pairs = [("temperature", w["temperature"], norm_row.temp_mean, norm_row.temp_std),
             ("humidity", w["humidity"], norm_row.hum_mean, norm_row.hum_std),
             ("wind_speed", w["wind_speed"], norm_row.wind_mean, norm_row.wind_std)]
    for name, val, mean, std in pairs:
        z = (val - mean) / std if std else 0.0
        out[name] = {"value": val, "mean": mean, "std": std, "z": z, "label": anomaly_label(z)}
    # дополнительно - температура относительно нормы ±7 дней вокруг сегодняшней даты
    zw = (w["temperature"] - win_mean) / win_std if win_std else 0.0
    out["temperature_window"] = {"mean": win_mean, "std": win_std, "z": zw, "label": anomaly_label(zw)}
    return out


# ----------------------------------------------------------------------------
# Этап 6. Рекомендации
# ----------------------------------------------------------------------------
def get_recommendations(city, season, w, comfort_score, anomalies):
    recommendations = []
    temp, humidity, wind = w["temperature"], w["humidity"], w["wind_speed"]

    # по нечёткой оценке комфорта
    if comfort_score < 35:
        recommendations.append(f"Низкий комфорт ({comfort_score:.0f}/100): сократите время на улице")
    elif comfort_score >= 65:
        recommendations.append(f"Погода комфортная ({comfort_score:.0f}/100): подходит для прогулок")

    # по городам (экспертные знания)
    if city == "Saint Petersburg":
        if humidity > 85 and temp < 5:
            recommendations.append("Сыро и холодно: непромокаемая одежда обязательна")
        if wind > 10:
            recommendations.append("Сильный ветер: в прибрежных районах ветер сильнее")
        if wind > 20:
            recommendations.append("Опасная скорость ветра (>20 м/с) в прибрежных зонах: избегайте набережных")
    elif city == "Moscow":
        if temp >= 23:
            recommendations.append("Жарко: при 3 и более днях подряд это волна жары; в центре меньше "
                                   "зелени и тепловой буфер слабее, ищите тень и пейте воду")
        if wind > 10:
            recommendations.append("Шквалистый ветер: осторожно рядом с деревьями и рекламными конструкциями")
    elif city == "Kazan":
        if season == "winter" and temp < -20:
            recommendations.append("Сильный мороз: защитите лицо и руки, ограничьте время на улице")
        if season == "winter" and wind < 3 and temp < 0:
            recommendations.append("Тихая морозная погода: утром в разных районах города "
                                   "температура может отличаться до 5°C")

    # общие
    if wind > 15:
        recommendations.append("Очень сильный ветер: уберите лёгкие вещи с балкона")
    if humidity < 30:
        recommendations.append("Низкая влажность: увлажняйте кожу, пейте воду")
    if humidity > 85:
        recommendations.append("Высокая влажность: возможны осадки")

    # по аномалиям
    tw = anomalies["temperature_window"]
    if abs(tw["z"]) >= 2:
        direction = "теплее" if tw["z"] > 0 else "холоднее"
        recommendations.append(f"Температура аномально отличается от нормы на эту дату "
                               f"(на {abs(w['temperature'] - tw['mean']):.1f}°C {direction}): "
                               f"подбирайте одежду по факту, а не по календарю")
    return recommendations


# ----------------------------------------------------------------------------
# Этап 7. Визуализация
# ----------------------------------------------------------------------------
def _finish(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def _plot_var(ax, var, title, current=None):
    for term in var.terms:
        ax.plot(var.universe, var[term].mf, linewidth=2,
                label=TERM_RU[var.label].get(term, term))
    if current is not None:
        ax.axvline(current, color="k", linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def plot_summary_table(df):
    fig, ax = plt.subplots(figsize=(14, 1.2 + 0.6 * len(df)))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    ax.set_title("Текущая погода и нечёткая оценка", fontsize=12)
    _finish(fig, "table_summary.png")


def plot_membership_functions(systems, current):
    """Сетка: города x (температура, влажность, ветер, комфорт)."""
    fig, axes = plt.subplots(len(systems), 4, figsize=(18, 3.6 * len(systems)), squeeze=False)
    for i, (city, fs) in enumerate(systems.items()):
        w = current[city]
        ru = CITIES[city]["ru"]
        _plot_var(axes[i][0], fs.temperature,
                  f"{ru}: температура ({SEASON_RU[fs.season]}), °C", w["temperature"])
        _plot_var(axes[i][1], fs.humidity, f"{ru}: влажность, %", w["humidity"])
        _plot_var(axes[i][2], fs.wind, f"{ru}: ветер, м/с", w["wind_speed"])
        _plot_var(axes[i][3], fs.comfort, f"{ru}: комфорт (выход)")
    _finish(fig, "membership_functions.png")


def plot_temperature_by_season(city="Moscow"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, season in zip(axes.ravel(), SEASONS):
        fs = FuzzyWeatherSystem(city, season)
        _plot_var(ax, fs.temperature, f"{CITIES[city]['ru']}: термы температуры, {SEASON_RU[season]}")
    _finish(fig, "temperature_terms_by_season.png")


def plot_seasonal_profiles(norms, current):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    specs = [("temp_mean", "temp_std", "temperature", "Температура, °C"),
             ("hum_mean", "hum_std", "humidity", "Влажность, %"),
             ("wind_mean", "wind_std", "wind_speed", "Ветер, м/с")]
    width = 0.8 / len(CITIES)
    for ax, (m, s, key, title) in zip(axes, specs):
        for k, (city, info) in enumerate(CITIES.items()):
            means = [norms.loc[(city, se), m] for se in SEASONS]
            stds = [norms.loc[(city, se), s] for se in SEASONS]
            xs = np.arange(len(SEASONS)) + k * width
            bars = ax.bar(xs, means, width, yerr=stds, capsize=3, label=info["ru"], alpha=0.85)
            if city in current:
                idx = SEASONS.index(get_season(datetime.now().month))
                ax.scatter(xs[idx], current[city][key], marker="*", s=180, color="k", zorder=5,
                           label="сейчас" if k == 0 else None)
        ax.set_xticks(np.arange(len(SEASONS)) + width)
        ax.set_xticklabels([SEASON_RU[s_] for s_ in SEASONS])
        ax.set_title(title + " (норма ± σ)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    _finish(fig, "seasonal_profiles.png")


def plot_trends(history):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
    current_year = date.today().year
    for city, df in history.items():
        ru = CITIES[city]["ru"]
        monthly = df["temperature"].resample("MS").mean()
        axes[0].plot(monthly.index, monthly.values, label=ru, linewidth=1.5)

        yearly = df[df.index.year < current_year]["temperature"].groupby(df[df.index.year < current_year].index.year).mean()
        line, = axes[1].plot(yearly.index, yearly.values, marker="o", label=ru)
        if len(yearly) >= 3:
            k, b = np.polyfit(yearly.index, yearly.values, 1)
            axes[1].plot(yearly.index, k * yearly.index + b, "--", color=line.get_color(),
                         label=f"тренд {ru}: {k:+.2f} °C/год")
    axes[0].set_title("Среднемесячная температура")
    axes[0].set_ylabel("°C")
    axes[1].set_title("Среднегодовая температура и линейный тренд")
    axes[1].set_xticks(sorted({y for df in history.values() for y in df.index.year if y < current_year}))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    _finish(fig, "trends.png")


# ----------------------------------------------------------------------------
# Основная программа
# ----------------------------------------------------------------------------
def main():
    now = datetime.now()
    season = get_season(now.month)
    print(f"Сегодня: {now:%d.%m.%Y}, сезон: {SEASON_RU[season]}\n")

    # Этап 1
    print("=== Этап 1. Текущая погода ===")
    current_data = {}
    for city in CITIES:
        data = get_current_weather(city, API_KEY)
        if data:
            current_data[city] = data
            print(f"{CITIES[city]['ru']}: {data['temperature']}°C "
                  f"(ощущается {data['feels_like']}°C), {data['description']}")
    if not current_data:
        print("Нет данных ни по одному городу: проверьте API-ключ и интернет.")
        return

    # Этап 2
    print("\n=== Этап 2. Исторические данные (Open-Meteo) ===")
    history = {}
    for city in CITIES:
        history[city] = get_history(city)
        h = history[city]
        print(f"{CITIES[city]['ru']}: {len(h)} суток, {h.index.min():%d.%m.%Y} - {h.index.max():%d.%m.%Y}")

    # Этап 3
    print("\n=== Этап 3. Климатические нормы по сезонам ===")
    norms = compute_seasonal_norms(history)
    print(norms.to_string())
    norms.to_csv(OUT_DIR / "seasonal_norms.csv", encoding="utf-8-sig")

    # Этапы 4-6 по каждому городу
    rows, systems, anomalies_all, recs_all = [], {}, {}, {}
    for city, w in current_data.items():
        ru = CITIES[city]["ru"]
        t, h, wind = w["temperature"], w["humidity"], w["wind_speed"]

        # Этап 4
        ling_t = numeric_to_linguistic(t, "temperature", city, season)
        ling_h = numeric_to_linguistic(h, "humidity", city, season)
        ling_w = numeric_to_linguistic(wind, "wind", city, season)

        # Этап 5
        fs = FuzzyWeatherSystem(city, season)
        systems[city] = fs
        deg = fs.memberships(t, h, wind)
        score, comfort_term = fs.evaluate(t, h, wind)

        # Сравнение с нормами
        win_mean, win_std = window_temp_norm(history[city], now)
        an = detect_anomalies(w, norms.loc[(city, season)], win_mean, win_std)
        anomalies_all[city] = an

        # Этап 6
        recs = get_recommendations(city, season, w, score, an)
        recs_all[city] = recs

        print(f"\n=== {ru} ===")
        print(f"Лингвистически: температура - {ling_t}; влажность - {ling_h}; ветер - {ling_w}")
        print("Фаззификация:")
        print("  температура:", format_memberships(deg, "temperature"))
        print("  влажность:  ", format_memberships(deg, "humidity"))
        print("  ветер:      ", format_memberships(deg, "wind_speed"))
        print(f"Нечёткий вывод: комфорт = {score:.1f}/100 ({TERM_RU['comfort'][comfort_term]}), "
              f"правил в базе: {len(fs.rules)}")
        print("Сравнение с сезонной нормой:")
        names = {"temperature": "температура", "humidity": "влажность", "wind_speed": "ветер"}
        for k in ("temperature", "humidity", "wind_speed"):
            a = an[k]
            print(f"  {names[k]}: {a['value']:.1f} (норма {a['mean']:.1f} ± {a['std']:.1f}, "
                  f"z={a['z']:+.2f}) - {a['label']}")
        aw = an["temperature_window"]
        print(f"  температура к норме на эту дату (±7 дн.): {aw['mean']:.1f} ± {aw['std']:.1f}, "
              f"z={aw['z']:+.2f} - {aw['label']}")
        print("Рекомендации:")
        for r in recs:
            print("  -", r)

        rows.append({
            "Город": ru,
            "Сезон": SEASON_RU[season],
            "T, °C": round(t, 1),
            "Ощущ., °C": round(w["feels_like"], 1),
            "Влажн., %": h,
            "Ветер, м/с": wind,
            "Описание": w["description"],
            "Температура (термин)": ling_t,
            "Комфорт, балл": round(score, 1),
            "Комфорт (терм)": TERM_RU["comfort"][comfort_term],
            "z (сезон)": round(an["temperature"]["z"], 2),
            "z (±7 дн.)": round(aw["z"], 2),
            "Статус": aw["label"],
        })

    # Этап 7
    print("\n=== Этап 7. Визуализация ===")
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    plot_summary_table(summary)
    plot_membership_functions(systems, current_data)
    plot_temperature_by_season("Moscow")
    plot_seasonal_profiles(norms, current_data)
    plot_trends(history)
    print(f"\nГрафики и таблицы сохранены в папку: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()


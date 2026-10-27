#получение текущих данных из OpenWeatherMap API
import requests
import time
from datetime import datetime

def get_current_weather(city, api_key):
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
        "lang":"ru"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        weather_info = {
            'city': city,
            'timestamp': datetime.now(),
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed'],
            'wind_gust': data['wind'].get('gust', 0),
            'description': data['weather'][0]['description']
        }
        return weather_info
    except Exception as e:
        print(f"Ошибка получения данных для {city}: {e}")
        return None
        
API_KEY = "074481e9fefb2838fcefc3cb2df3f60e"
cities = ["Kazan", "Moscow", "Saint Petersburg"]

current_data = {}
for city in cities:
    data = get_current_weather(city, API_KEY)
    if data:
        current_data[city] = data
        print(f"{city}: {data['temperature']}ºC, {data['description']}")
    

#расчет климатических норм по сезонам
def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"
        
#преобразование чисел в линвистические значения
def numeric_to_linguistic(value, var_name, city_name, season):
    if var_name == 'temperature':
        if city_name == "Kazan":
            if season == "winter":
                if value < -25: return "экстремально холодно"
                elif value < -15: return "холодно"
                elif value < -5: return "умеренно холодно"
                elif value < 0: return "близко к нулю"
                else: return "тепло для зимы"
            elif season == "summer":
                if value < 15: return "холодно для лета"
                elif value < 20: return "прохладно"
                elif value < 25: return "комфортно"
                elif value < 30: return "тепло"
                else: return "жарко"
        elif city_name == "Saint Petersburg":
            if season == "winter":
                if value < -15: return "морозно (редкость)"
                elif value < -5: return "холодно"
                elif value < 0: return "около нуля"
                else: return "оттепель"
            elif season == "summer":
                if value < 10: return "очень холодно"
                elif value < 15: return "прохладно"
                elif value < 20: return "умеренно"
                
#реализация нечеткой системы с учетом сезонов
self.temperature = ctrl.Antecedent(np.arange(-30, 40, 1), 'temperature')
self.humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
self.wind = ctrl.Antecedent(np.arange(0, 30, 1), 'wind')

self.comfort = ctrl.Consequent(np.arange(0, 101, 1), 'comfort')

#формирование реккомендаций
if self.season == "winter":
    self.temperature['extremely_cold'] = fuzz.trimf(self.temperature.universe, [-30, -25, -20])
    self.temperature['cold'] = fuzz.trimf(self.temperature.universe, [-20, -15, -10])
    self.temperature['normal'] = fuzz.trimf(self.temperature.universe, [mean_temp-std_temp, mean_temp+std_temp])
    self.temperature['warm_for_winter'] = fuzz.trimf(self.temperature.universe, [10, 15, 20])
elif self.season == "summer":
    self.temperature['cool'] = fuzz.trimf(self.temperature.universe, [10, 15, 20])
    self.temperature['normal'] = fuzz.trimf(self.temperature.universe, [mean_temp-std_temp, mean_temp+std_temp])
    self.temperature['warm'] = fuzz.trimf(self.temperature.universe, [23, 27, 30])
    self.temperature['hot'] = fuzz.trimf(self.temperature.universe, [28, 32, 38])
    
if self.city == "Saint Petersburg":
    rules.append(ctrl.Rule(
        self.humidity['high']&self.wind['windy'],
        self.comfort['bad']
    ))
    
self.system = ctrl.ControlSystem(rules)
self.simulation = ctrl.ControlSystemSimulation(self.system)

elif city == "Saint Petersburg":
    if humidity > 85 and temp < 5:
        recommendations.append("Сыро и холодно: непромокаемая одежда обязательна")
    if wind > 10:
        recommendations.append("Сильный ветер: в прибрежных районах ветер сильнее")
        
#общие рекомендации
if wind > 15:
    recommendations.append("Очень сильный ветер: уберите легкие вещи с балкона")
if humidity < 30:
    recommendations.append("Низкая влажность: увлажняйте кожу. пейте воду")
if humidity > 85:
    recommendations.append("Высокая влажность: возможны осадки")

return recommendations
        

import requests

BASE_URL = "https://api.openweathermap.org/data/2.5"

def get_current_weather(city, api_key):
    """
    Fetch current weather for a city.
    Returns dict with temperature, humidity,
    weather condition, and estimated rainfall.
    """
    try:
        # Current weather endpoint
        url    = f"{BASE_URL}/weather"
        params = {
            'q'     : city,
            'appid' : api_key,
            'units' : 'metric'
        }
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Extract weather info
            temp      = data['main']['temp']
            humidity  = data['main']['humidity']
            condition = data['weather'][0]['main']
            desc      = data['weather'][0]['description']
            city_name = data['name']
            country   = data['sys']['country']

            # Rainfall estimation from weather condition
            # OpenWeatherMap free tier doesn't give annual rainfall
            # We estimate from current condition
            rain_1h = 0
            if 'rain' in data:
                rain_1h = data['rain'].get('1h', 0)

            # Map condition to seasonal rainfall estimate
            condition_rainfall_map = {
                'Thunderstorm' : 800,
                'Drizzle'      : 400,
                'Rain'         : 650,
                'Snow'         : 300,
                'Clear'        : 250,
                'Clouds'       : 450,
                'Mist'         : 350,
                'Fog'          : 350,
                'Haze'         : 300,
                'Windy'        : 300,
                'Stormy'       : 750,
            }
            est_rainfall = condition_rainfall_map.get(
                condition, 400
            )

            # Map API condition to our app's weather options
            condition_map = {
                'Thunderstorm' : 'Stormy',
                'Drizzle'      : 'Rainy',
                'Rain'         : 'Rainy',
                'Snow'         : 'Cloudy',
                'Clear'        : 'Sunny',
                'Clouds'       : 'Cloudy',
                'Mist'         : 'Cloudy',
                'Fog'          : 'Cloudy',
                'Haze'         : 'Cloudy',
                'Windy'        : 'Windy',
                'Stormy'       : 'Stormy',
            }
            app_condition = condition_map.get(
                condition, 'Sunny'
            )

            return {
                'success'       : True,
                'city'          : city_name,
                'country'       : country,
                'temperature'   : round(temp, 1),
                'humidity'      : round(humidity, 1),
                'condition'     : app_condition,
                'description'   : desc.title(),
                'est_rainfall'  : est_rainfall,
                'rain_1h'       : rain_1h,
                'raw_condition' : condition,
            }

        elif response.status_code == 404:
            return {
                'success' : False,
                'error'   : f"City '{city}' not found. Check spelling."
            }
        elif response.status_code == 401:
            return {
                'success' : False,
                'error'   : "Invalid API key. Check your OpenWeatherMap key."
            }
        else:
            return {
                'success' : False,
                'error'   : f"API error: {response.status_code}"
            }

    except requests.exceptions.Timeout:
        return {
            'success' : False,
            'error'   : "Request timed out. Check internet connection."
        }
    except requests.exceptions.ConnectionError:
        return {
            'success' : False,
            'error'   : "No internet connection."
        }
    except Exception as e:
        return {
            'success' : False,
            'error'   : f"Unexpected error: {str(e)}"
        }


def get_weather_farming_advice(weather_data):
    """
    Returns farming advice based on current weather.
    """
    if not weather_data['success']:
        return []

    advice  = []
    temp    = weather_data['temperature']
    cond    = weather_data['raw_condition']
    humid   = weather_data['humidity']

    if temp > 35:
        advice.append({
            'type'    : 'warning',
            'message' : f"🌡️ High temperature ({temp}°C) — "
                        f"avoid midday irrigation, "
                        f"risk of heat stress for crops"
        })
    elif temp < 10:
        advice.append({
            'type'    : 'warning',
            'message' : f"🥶 Low temperature ({temp}°C) — "
                        f"frost risk! Protect sensitive crops"
        })
    else:
        advice.append({
            'type'    : 'success',
            'message' : f"✅ Temperature ({temp}°C) is suitable "
                        f"for most crops"
        })

    if cond in ['Rain', 'Thunderstorm', 'Drizzle']:
        advice.append({
            'type'    : 'info',
            'message' : "🌧️ Rain detected — skip irrigation today, "
                        "delay fertilizer application to avoid runoff"
        })
    elif cond == 'Clear' and temp > 30:
        advice.append({
            'type'    : 'warning',
            'message' : "☀️ Hot and sunny — increase irrigation frequency"
        })

    if humid > 85:
        advice.append({
            'type'    : 'warning',
            'message' : f"💧 High humidity ({humid}%) — "
                        f"watch for fungal diseases"
        })
    elif humid < 30:
        advice.append({
            'type'    : 'warning',
            'message' : f"🏜️ Low humidity ({humid}%) — "
                        f"crops may need more water"
        })

    return advice


if __name__ == "__main__":
    # Quick test — replace with your API key
    API_KEY = "your_api_key_here"
    result  = get_current_weather("Chennai", API_KEY)
    print(result)
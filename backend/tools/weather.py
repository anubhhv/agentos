import os
import httpx

OW_KEY = os.getenv("OPENWEATHER_API_KEY", "")
BASE = "https://api.openweathermap.org/data/2.5"
GEO = "https://api.openweathermap.org/geo/1.0"


async def get_weather(location: str, units: str = "metric") -> dict:
    """Get current weather + 5-day forecast for any city."""
    if not OW_KEY:
        return {"error": "OPENWEATHER_API_KEY not set"}

    unit_label = "°C" if units == "metric" else "°F"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Geocode location
            geo_r = await client.get(f"{GEO}/direct", params={
                "q": location, "limit": 1, "appid": OW_KEY
            })
            geo_r.raise_for_status()
            geo = geo_r.json()
            if not geo:
                return {"error": f"Location '{location}' not found"}

            lat, lon = geo[0]["lat"], geo[0]["lon"]
            city_name = geo[0].get("name", location)
            country = geo[0].get("country", "")

            # Current weather
            cur_r = await client.get(f"{BASE}/weather", params={
                "lat": lat, "lon": lon, "appid": OW_KEY,
                "units": units
            })
            cur_r.raise_for_status()
            cur = cur_r.json()

            # 5-day forecast
            fc_r = await client.get(f"{BASE}/forecast", params={
                "lat": lat, "lon": lon, "appid": OW_KEY,
                "units": units, "cnt": 40
            })
            fc_r.raise_for_status()
            fc = fc_r.json()

        # Parse current
        current = {
            "temp": f"{cur['main']['temp']}{unit_label}",
            "feels_like": f"{cur['main']['feels_like']}{unit_label}",
            "humidity": f"{cur['main']['humidity']}%",
            "description": cur["weather"][0]["description"].capitalize(),
            "wind_speed": f"{cur['wind']['speed']} m/s",
            "visibility": f"{cur.get('visibility', 0) // 1000} km",
            "pressure": f"{cur['main']['pressure']} hPa",
        }

        # Parse forecast — daily summary (noon readings)
        daily = {}
        for item in fc["list"]:
            day = item["dt_txt"].split(" ")[0]
            if "12:00:00" in item["dt_txt"]:
                daily[day] = {
                    "date": day,
                    "temp": f"{item['main']['temp']}{unit_label}",
                    "description": item["weather"][0]["description"].capitalize(),
                    "humidity": f"{item['main']['humidity']}%",
                    "wind": f"{item['wind']['speed']} m/s"
                }

        return {
            "location": f"{city_name}, {country}",
            "coordinates": {"lat": lat, "lon": lon},
            "current": current,
            "forecast": list(daily.values())[:5],
            "units": units
        }

    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}

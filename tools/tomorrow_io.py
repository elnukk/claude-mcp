import os
import httpx
from fastapi import HTTPException

API_KEY = os.getenv("TOMORROW_IO_API_KEY")
BASE_URL = "https://api.tomorrow.io/v4"

async def get_tomorrow_weather(location: str, fields: str = "temperature,weatherCode", timesteps: str = "1h", units: str = "metric"):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Tomorrow.io API key not configured")

    params = {
        "location": location,
        "fields": fields.split(','),
        "timesteps": timesteps.split(','),
        "units": units,
        "apikey": API_KEY,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/weather/forecast", params=params)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching data from Tomorrow.io")

    return response.json()

async def get_weather_alerts(location: str):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Tomorrow.io API key not configured")

    params = {
        "location": location,
        "apikey": API_KEY,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/weather/alerts", params=params)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching data from Tomorrow.io")

    return response.json()

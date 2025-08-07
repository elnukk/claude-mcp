import os
import httpx
from fastapi import HTTPException

API_KEY = os.getenv("GOOGLE_WEATHER_API_KEY")
BASE_URL = "https://weather.googleapis.com/v1/currentConditions:lookup"

async def get_google_weather_current_conditions(latitude: float, longitude: float, api_key: str):
    if not api_key:
        raise HTTPException(status_code=500, detail="Google Weather API key not configured")

    params = {
        "key": api_key,
        "location.latitude": latitude,
        "location.longitude": longitude
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(BASE_URL, params=params)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching data from Google Weather API")

    return response.json()
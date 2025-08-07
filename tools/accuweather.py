import os
import httpx
from fastapi import HTTPException

API_KEY = os.getenv("ACCUWEATHER_API_KEY")
BASE_URL = "http://dataservice.accuweather.com/currentconditions/v1"

async def get_accuweather_current_conditions(location_key: str, api_key: str):
    if not api_key:
        raise HTTPException(status_code=500, detail="AccuWeather API key not configured")

    params = {
        "apikey": api_key,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/{location_key}", params=params)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching data from AccuWeather")

    return response.json()
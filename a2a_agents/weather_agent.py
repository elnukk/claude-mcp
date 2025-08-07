import asyncio
from src.mcp_weather_server.tools import open_meteo

class WeatherAgent:
    async def get_weather_forecast(self, latitude, longitude):
        print("Fetching 7-day weather forecast...")
        weather_forecast_response = await open_meteo.get_weather_forecast(latitude=latitude, longitude=longitude)
        if "error" in weather_forecast_response:
            print(f"Error: {weather_forecast_response['error']}")
            return None
        print("Weather forecast received.")
        return weather_forecast_response

async def main():
    agent = WeatherAgent()
    # Example usage (Patna, Bihar)
    lat = 25.6
    lon = 85.1
    forecast = await agent.get_weather_forecast(latitude=lat, longitude=lon)
    if forecast:
        print("\n--- Weather Forecast ---")
        print(forecast)
        print("----------------------")

if __name__ == "__main__":
    asyncio.run(main())

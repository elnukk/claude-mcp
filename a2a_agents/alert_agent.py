import asyncio
from src.mcp_weather_server.tools import alert_generation_tools
from unittest.mock import patch

class AlertAgent:
    async def generate_alert(self, crop, weather_data, growth_stage, latitude, longitude):
        with patch('src.mcp_weather_server.tools.openai_llm.predict_weather_alert') as mock_predict_weather_alert:
            # Mock the alert generation
            mock_predict_weather_alert.return_value = {
                "alert": "Heavy rainfall expected",
                "impact": "High risk of waterlogging in fields.",
                "recommendations": "Ensure proper drainage in fields."
            }

            print("Generating weather alert...")
            api_key = "test_api_key"  # This will be mocked, so the value doesn't matter
            alert_response = await alert_generation_tools.generate_weather_alert(
                crop=crop,
                weather_data=weather_data,
                growth_stage=growth_stage,
                api_key=api_key,
                latitude=latitude,
                longitude=longitude
            )
            return alert_response

async def main():
    agent = AlertAgent()
    # Example usage
    crop = "Wheat"
    growth_stage = "Flowering"
    lat = 25.6
    lon = 85.1
    # Mock weather data for the example
    weather_data = {
        "daily": {
            "time": ["2024-02-15", "2024-02-16"],
            "temperature_2m_max": [25, 26],
            "temperature_2m_min": [12, 13],
            "precipitation_sum": [0, 5]
        }
    }

    alert = await agent.generate_alert(
        crop=crop,
        weather_data=weather_data,
        growth_stage=growth_stage,
        latitude=lat,
        longitude=lon
    )

    if alert and "error" not in alert:
        print("\n--- Generated Weather Alert ---")
        print(f"Alert: {alert['alert']}")
        print(f"Impact: {alert['impact']}")
        print(f"Recommendations: {alert['recommendations']}")
        print("-----------------------------")
    elif alert and "error" in alert:
        print(f"Error generating alert: {alert['error']}")


if __name__ == "__main__":
    asyncio.run(main())

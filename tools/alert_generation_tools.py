"""
This module contains tools for alert generation.
"""
from . import openai_llm

async def generate_weather_alert(crop: str, weather_data: dict, growth_stage: str, api_key: str, latitude: float, longitude: float):
    """
    Generates a weather alert using an LLM.
    """
    prompt = f"Generate a weather alert for {crop} at the {growth_stage} stage. Weather data: {weather_data}"
    response = await openai_llm.predict_weather_alert(latitude=latitude, longitude=longitude, api_key=api_key)
    return response

async def prioritize_alerts(alerts_list: list, urgency_factors: dict):
    """
    Prioritizes a list of alerts based on urgency factors.
    """
    # Mock implementation
    for alert in alerts_list:
        alert["urgency"] = urgency_factors.get(alert["crop"], 0)
    return sorted(alerts_list, key=lambda x: x["urgency"], reverse=True)

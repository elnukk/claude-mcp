import os
import json
from fastapi import HTTPException
from openai import OpenAI
from . import open_meteo

async def predict_weather_alert(latitude: float, longitude: float, api_key: str):
    """
    Predicts weather alerts for a given location and crops using an OpenAI LLM.

    Args:
        latitude: The latitude of the location.
        longitude: The longitude of the location.
        crops: A list of crops to consider for the prediction.

    Returns:
        A dictionary containing the predicted weather alert.
    """
    try:
        weather_data = await open_meteo.get_weather_forecast(latitude, longitude)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=f"Error getting weather data: {e.detail}")

    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""
        Given the following weather data for a location:
        {weather_data}

        Please predict any potential weather alerts for these crops in the next 7 days.
        For the given region, consider what crops are possible to grow and their sensitivity to weather conditions.
        Include the following details in your response:
        - Expected weather conditions (e.g., temperature, precipitation, wind speed)
        - Potential weather alerts (e.g., frost, drought, heavy rainfall)
        - Impact on crops (e.g., growth, yield, disease risk)
        - Recommended actions for farmers (e.g., irrigation, protection measures)
        - Any other relevant information that could help farmers prepare for the weather conditions.
        Provide a summary of the potential impact on the crops and any recommended actions.
        Format your response as a JSON object with the following structure:
        {{
            "alert": "Description of the alert",
            "impact": "Description of the impact on crops",
            "recommendations": "Recommended actions for farmers"
        }}
        Do not include any additional text outside of the JSON object. no line changes or markdown formatting.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that predicts weather alerts for farmers."},
                {"role": "user", "content": prompt}
            ],
            response_format= { "type": "json_object" }
        )

        response = response.choices[0].message.content
        if response:
            return json.loads(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting prediction from OpenAI: {str(e)}")

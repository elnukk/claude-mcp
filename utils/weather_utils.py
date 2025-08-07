TOOL_CONFIG = {
    "get_weather_forecast": {"module": "open_meteo"},
    "get_current_weather": {"module": "open_meteo"},
    "get_historical_weather": {"module": "open_meteo"},
    "analyze_weather_trends": {"module": "open_meteo"},
    "get_tomorrow_weather": {"module": "tomorrow_io"},
    "get_weather_alerts": {"module": "tomorrow_io"},
    "get_google_weather_current_conditions": {"module": "google_weather"},
    "get_openweathermap_weather": {"module": "openweathermap"},
    "get_accuweather_current_conditions": {"module": "accuweather"},
    "predict_weather_alert": {"module": "openai_llm"},
    "list_villages": {"module": "geographic_tools"},
    "reverse_geocode": {"module": "geographic_tools"},
    "get_administrative_bounds": {"module": "geographic_tools"},
    "get_crop_calendar": {"module": "crop_calendar_tools"},
    "get_prominent_crops": {"module": "crop_calendar_tools"},
    "estimate_crop_stage": {"module": "crop_calendar_tools"},
    "generate_weather_alert": {"module": "alert_generation_tools"},
    "prioritize_alerts": {"module": "alert_generation_tools"},
}

def get_tool_config(tool_name: str):
    return TOOL_CONFIG.get(tool_name)

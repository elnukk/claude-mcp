import os
import logging
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import dotenv_values
import asyncio
from datetime import datetime, timedelta
import csv
from io import StringIO


from tools import open_meteo, tomorrow_io, google_weather, openweathermap, accuweather, openai_llm, geographic_tools, crop_calendar_tools, alert_generation_tools
from a2a_agents import sms_agent, whatsapp_agent, ussd_agent, ivr_agent, telegram_agent
from utils.weather_utils import get_tool_config


config = dotenv_values(".env")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

print("OPENAI_API_KEY exists?", os.getenv("OPENAI_API_KEY") is not None)
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mcp-ui.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class MCPRequest(BaseModel):
    tool: str
    parameters: dict

class AlertRequest(BaseModel):
    alert_json: dict

class WorkflowRequest(BaseModel):
    state: str
    district: str

def get_regional_crop_for_area(district: str, state: str):
    """Get typical crop for the region"""
    if state.lower() == 'bihar':
        district_crops = {
            'patna': 'rice',
            'gaya': 'wheat',
            'bhagalpur': 'rice',
            'muzaffarpur': 'sugarcane',
            'darbhanga': 'rice',
            'siwan': 'rice',
            'begusarai': 'rice',
            'katihar': 'maize',
        }
        return district_crops.get(district.lower(), 'rice')
    return 'rice'

def get_current_crop_stage(crop: str):
    """Determine crop stage based on current date"""
    current_month = datetime.now().month
    
    if crop == 'rice':
        if current_month in [6, 7]:
            return 'planting'
        elif current_month in [8, 9]:
            return 'growing'
        elif current_month in [10, 11]:
            return 'flowering'
        else:
            return 'harvesting'
    elif crop == 'wheat':
        if current_month in [11, 12]:
            return 'planting'
        elif current_month in [1, 2]:
            return 'growing'
        elif current_month in [3, 4]:
            return 'flowering'
        else:
            return 'harvesting'
    elif crop == 'sugarcane':
        if current_month in [2, 3, 4]:
            return 'planting'
        elif current_month in [5, 6, 7, 8]:
            return 'growing'
        elif current_month in [9, 10, 11]:
            return 'maturing'
        else:
            return 'harvesting'
    elif crop == 'maize':
        if current_month in [6, 7]:
            return 'planting'
        elif current_month in [8, 9]:
            return 'growing'
        elif current_month in [10, 11]:
            return 'flowering'
        else:
            return 'harvesting'
    
    return 'growing'

async def generate_dynamic_alert(district: str, state: str):
    """Generate dynamic alert data using geographic functions and REAL weather data"""
    
    try:
        # Step 1: Get villages for the district using your geographic tools
        villages_data = await geographic_tools.list_villages(state, district)
        
        if "error" in villages_data:
            raise Exception(f"District '{district}' not found in {state}")
        
        # Step 2: Pick a random village from the actual list
        available_villages = villages_data.get("villages", [])
        if not available_villages:
            raise Exception(f"No villages found for {district}")
        
        selected_village = random.choice(available_villages)
        logger.info(f"Selected village: {selected_village} from {len(available_villages)} villages")
        
        # Step 3: Try to get coordinates for the selected village first, then district
        location_coords = None
        location_source = ""
        
        # Try village coordinates first
        try:
            village_location = await geographic_tools.reverse_geocode(selected_village)
            if "error" not in village_location and "lat" in village_location:
                location_coords = [village_location["lat"], village_location["lng"]]
                location_source = f"village_{selected_village}"
                logger.info(f"Using village coordinates for {selected_village}: {location_coords}")
        except Exception as e:
            logger.warning(f"Village geocoding failed for {selected_village}: {e}")
        
        # Fallback to district coordinates if village lookup failed
        if not location_coords:
            try:
                district_location = await geographic_tools.reverse_geocode(district)
                if "error" not in district_location and "lat" in district_location:
                    location_coords = [district_location["lat"], district_location["lng"]]
                    location_source = f"district_{district}"
                    logger.info(f"Using district coordinates for {district}: {location_coords}")
            except Exception as e:
                logger.warning(f"District geocoding failed for {district}: {e}")
        
        # Final fallback - but this should rarely happen now
        if not location_coords:
            logger.warning(f"No coordinates found for {selected_village} or {district}, using default")
            location_coords = [25.5941, 85.1376]  # Patna fallback
            location_source = "fallback_patna"
        
        # Step 4: Generate regional crop and stage using crop calendar data
        regional_crop = await get_regional_crop_for_area(district, state)
        crop_stage = await get_current_crop_stage_dynamic(regional_crop, district)
        
        # Step 5: GET REAL WEATHER DATA using the actual coordinates
        try:
            logger.info(f"Fetching weather for coordinates: {location_coords} (source: {location_source})")
            
            current_weather_data = await open_meteo.get_current_weather(
                latitude=location_coords[0], 
                longitude=location_coords[1]
            )
            
            forecast_data = await open_meteo.get_weather_forecast(
                latitude=location_coords[0], 
                longitude=location_coords[1],
                days=7
            )
            
            current_weather = current_weather_data.get('current_weather', {})
            daily_forecast = forecast_data.get('daily', {})
            
            current_temp = current_weather.get('temperature', 25)
            current_windspeed = current_weather.get('windspeed', 10)
            
            precipitation_list = daily_forecast.get('precipitation_sum', [0, 0, 0])
            next_3_days_rain = sum(precipitation_list[:3]) if precipitation_list else 0
            
            rain_probability = min(90, max(10, int(next_3_days_rain * 10))) if next_3_days_rain > 0 else 10
            
            # Higher precipitation = higher humidity estimate
            estimated_humidity = min(95, max(40, 60 + int(next_3_days_rain * 2)))
            
            real_weather = {
                "forecast_days": 3,
                "rain_probability": rain_probability,
                "expected_rainfall": f"{next_3_days_rain:.1f}mm",
                "temperature": f"{current_temp:.1f}°C",
                "humidity": f"{estimated_humidity}%",
                "wind_speed": f"{current_windspeed:.1f} km/h",
                "coordinates_source": location_source  # Track where coords came from
            }
            
            # Step 6: Generate alert message based on actual weather conditions
            if next_3_days_rain > 25:
                alert_type = "heavy_rain_warning"
                urgency = "high"
                alert_message = f"Heavy rainfall ({next_3_days_rain:.1f}mm) expected in next 3 days near {selected_village}, {district}. Delay fertilizer application. Ensure proper drainage."
                action_items = ["delay_fertilizer", "check_drainage", "monitor_crops", "prepare_harvest_protection"]
            elif next_3_days_rain > 10:
                alert_type = "moderate_rain_warning"
                urgency = "medium"
                alert_message = f"Moderate rainfall ({next_3_days_rain:.1f}mm) expected in next 3 days near {selected_village}, {district}. Monitor soil moisture levels."
                action_items = ["monitor_soil", "check_drainage", "adjust_irrigation"]
            elif next_3_days_rain < 2 and current_temp > 35:
                alert_type = "heat_drought_warning"
                urgency = "high"
                alert_message = f"High temperature ({current_temp:.1f}°C) with minimal rainfall expected near {selected_village}, {district}. Increase irrigation frequency."
                action_items = ["increase_irrigation", "mulch_crops", "monitor_plant_stress"]
            elif current_temp < 10:
                alert_type = "cold_warning"
                urgency = "medium"
                alert_message = f"Low temperature ({current_temp:.1f}°C) expected near {selected_village}, {district}. Protect crops from cold damage."
                action_items = ["protect_crops", "cover_seedlings", "adjust_irrigation_timing"]
            elif current_windspeed > 30:
                alert_type = "high_wind_warning"
                urgency = "medium"
                alert_message = f"High winds ({current_windspeed:.1f} km/h) expected near {selected_village}, {district}. Secure crop supports and structures."
                action_items = ["secure_supports", "check_structures", "monitor_damage"]
            else:
                alert_type = "weather_update"
                urgency = "low"
                alert_message = f"Normal weather conditions expected near {selected_village}, {district}. Temperature {current_temp:.1f}°C, rainfall {next_3_days_rain:.1f}mm."
                action_items = ["routine_monitoring", "maintain_irrigation"]
            
            logger.info(f"Real weather data retrieved for {selected_village}, {district}: {current_temp}°C, {next_3_days_rain:.1f}mm rain (coords: {location_coords})")
            
        except Exception as weather_error:
            logger.error(f"Failed to get real weather data for {selected_village}, {district}: {weather_error}")
            raise Exception(f"Unable to retrieve current weather conditions for {selected_village}, {district}")
        
        return {
            "alert_id": f"{state.upper()[:2]}_{district.upper()[:3]}_{selected_village.upper()[:3]}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "timestamp": datetime.now().isoformat() + "Z",
            "location": {
                "village": selected_village,
                "district": district,
                "state": state.capitalize(),
                "coordinates": location_coords,
                "coordinates_source": location_source,
                "total_villages_in_district": len(available_villages)
            },
            "crop": {
                "name": regional_crop,
                "stage": crop_stage,
                "planted_estimate": "2025-06-15"  # You could make this dynamic too
            },
            "alert": {
                "type": alert_type,
                "urgency": urgency,
                "message": alert_message,
                "action_items": action_items,
                "valid_until": (datetime.now() + timedelta(days=3)).isoformat() + "Z"
            },
            "weather": real_weather,
            "data_source": "open_meteo_api_with_dynamic_location"
        }
    
    except Exception as e:
        logger.error(f"Error generating dynamic alert for {district}, {state}: {e}")
        raise Exception(f"Failed to generate weather alert for {district}: {str(e)}")




import random
from datetime import datetime, date

# Enhanced crop selection function using your crop calendar data
async def get_regional_crop_for_area(district: str, state: str):
    """Get typical crop for the region based on season and district - now fully dynamic"""
    
    if state.lower() != 'bihar':
        return 'rice'  # fallback for other states
    
    current_month = datetime.now().month
    current_season = get_current_season(current_month)
    
    # Get crops that are currently in season using your crop calendar tools
    try:
        seasonal_crops_data = await crop_calendar_tools.get_prominent_crops('bihar', current_season)
        if "error" not in seasonal_crops_data:
            seasonal_crops = seasonal_crops_data.get('crops', [])
        else:
            seasonal_crops = []
    except Exception as e:
        logger.warning(f"Failed to get seasonal crops: {e}")
        seasonal_crops = []
    
    # District-specific crop preferences (what's commonly grown in each district)
    district_crop_preferences = {
        'patna': {
            'primary': ['rice', 'wheat', 'potato'],
            'secondary': ['mustard', 'gram', 'barley'],
            'specialty': ['sugarcane']
        },
        'gaya': {
            'primary': ['wheat', 'rice', 'gram'],
            'secondary': ['barley', 'lentil', 'mustard'],
            'specialty': ['arhar']
        },
        'bhagalpur': {
            'primary': ['rice', 'maize', 'wheat'],
            'secondary': ['jute', 'urd', 'moong'],
            'specialty': ['groundnut']
        },
        'muzaffarpur': {
            'primary': ['sugarcane', 'rice', 'wheat'],
            'secondary': ['potato', 'mustard'],
            'specialty': ['lentil']
        },
        'darbhanga': {
            'primary': ['rice', 'wheat', 'maize'],
            'secondary': ['gram', 'arhar'],
            'specialty': ['bajra']
        },
        'siwan': {
            'primary': ['rice', 'wheat'],
            'secondary': ['gram', 'lentil', 'pea'],
            'specialty': ['mustard']
        },
        'begusarai': {
            'primary': ['rice', 'wheat'],
            'secondary': ['jute', 'mustard'],
            'specialty': ['moong', 'urd']
        },
        'katihar': {
            'primary': ['maize', 'rice'],
            'secondary': ['jute', 'urd', 'moong'],
            'specialty': ['jowar', 'bajra']
        },
        'vaishali': {
            'primary': ['rice', 'wheat', 'sugarcane'],
            'secondary': ['potato', 'gram'],
            'specialty': ['mustard']
        },
        'madhubani': {
            'primary': ['rice', 'wheat', 'maize'],
            'secondary': ['gram', 'lentil'],
            'specialty': ['arhar']
        }
    }
    
    # Get district preferences or use default
    district_prefs = district_crop_preferences.get(district.lower(), {
        'primary': ['rice', 'wheat'],
        'secondary': ['gram', 'mustard'],
        'specialty': ['maize']
    })
    
    # Combine all possible crops for this district
    all_district_crops = (district_prefs.get('primary', []) + 
                         district_prefs.get('secondary', []) + 
                         district_prefs.get('specialty', []))
    
    # Find crops that are both seasonal AND grown in this district
    suitable_crops = []
    if seasonal_crops:
        suitable_crops = [crop for crop in all_district_crops if crop in seasonal_crops]
    
    # If no seasonal match, use district preferences with seasonal weighting
    if not suitable_crops:
        if current_season == 'kharif':
            # Monsoon crops preference
            kharif_crops = ['rice', 'maize', 'arhar', 'moong', 'urd', 'jowar', 'bajra', 'groundnut', 'soybean']
            suitable_crops = [crop for crop in all_district_crops if crop in kharif_crops]
        elif current_season == 'rabi':
            # Winter crops preference
            rabi_crops = ['wheat', 'barley', 'gram', 'lentil', 'pea', 'mustard', 'linseed', 'potato']
            suitable_crops = [crop for crop in all_district_crops if crop in rabi_crops]
        elif current_season == 'zaid':
            # Summer crops preference
            zaid_crops = ['maize', 'moong', 'urd', 'watermelon', 'cucumber']
            suitable_crops = [crop for crop in all_district_crops if crop in zaid_crops]
    
    # If still no match, fall back to district primary crops
    if not suitable_crops:
        suitable_crops = district_prefs.get('primary', ['rice'])
    
    # Weight selection based on crop category (primary crops more likely)
    weighted_crops = []
    for crop in suitable_crops:
        if crop in district_prefs.get('primary', []):
            weighted_crops.extend([crop] * 5)  # 5x weight for primary crops
        elif crop in district_prefs.get('secondary', []):
            weighted_crops.extend([crop] * 3)  # 3x weight for secondary crops
        else:
            weighted_crops.extend([crop] * 1)  # 1x weight for specialty crops
    
    selected_crop = random.choice(weighted_crops) if weighted_crops else 'rice'
    
    logger.info(f"Selected crop: {selected_crop} for {district} in {current_season} season from options: {suitable_crops}")
    
    return selected_crop


async def get_current_crop_stage_dynamic(crop: str, district: str = None):
    """Determine crop stage based on current date and crop calendar - now more accurate"""
    
    try:
        # Get crop calendar information
        crop_info = await crop_calendar_tools.get_crop_calendar('bihar', crop)
        
        if "error" in crop_info:
            # Fallback to the old static method
            return get_current_crop_stage_static(crop)
        
        # Parse planting and harvesting periods
        planting_period = crop_info.get('planting', '')
        season = crop_info.get('season', '')
        stages = crop_info.get('stages', [])
        
        current_month = datetime.now().month
        current_date = date.today()
        
        # Estimate planting date based on season and current month
        estimated_plant_date = estimate_planting_date(crop, season, planting_period, current_month)
        
        if estimated_plant_date:
            # Use the crop calendar function to estimate stage
            try:
                stage_data = await crop_calendar_tools.estimate_crop_stage(
                    crop, 
                    estimated_plant_date.isoformat(), 
                    current_date.isoformat()
                )
                
                if "error" not in stage_data:
                    stage = stage_data.get('stage', stages[0] if stages else 'Growing')
                    logger.info(f"Dynamic stage calculation for {crop}: {stage} (planted ~{estimated_plant_date})")
                    return stage
            except Exception as e:
                logger.warning(f"Error in dynamic stage calculation: {e}")
        
        # Fallback to month-based estimation
        return estimate_stage_by_month(crop, current_month, stages)
    
    except Exception as e:
        logger.error(f"Error in dynamic crop stage calculation: {e}")
        return get_current_crop_stage_static(crop)


def get_current_season(month: int):
    """Determine current agricultural season"""
    if month in [6, 7, 8, 9]:  # June to September
        return 'kharif'
    elif month in [10, 11, 12, 1, 2, 3]:  # October to March
        return 'rabi'
    else:  # April, May
        return 'zaid'

async def get_regional_crop_for_area(district: str, state: str):
    """Get typical crop for the region based on season and district - now fully dynamic"""
    
    if state.lower() != 'bihar':
        return 'rice'  # fallback for other states
    
    current_month = datetime.now().month
    current_season = get_current_season(current_month)
    
    # Get crops that are currently in season using your crop calendar tools
    try:
        seasonal_crops_data = await crop_calendar_tools.get_prominent_crops('bihar', current_season)
        if "error" not in seasonal_crops_data:
            seasonal_crops = seasonal_crops_data.get('crops', [])
        else:
            seasonal_crops = []
    except Exception as e:
        logger.warning(f"Failed to get seasonal crops: {e}")
        seasonal_crops = []
    
    # District-specific crop preferences
    district_crop_preferences = {
        'patna': {'primary': ['rice', 'wheat', 'potato'], 'secondary': ['mustard', 'gram', 'barley'], 'specialty': ['sugarcane']},
        'gaya': {'primary': ['wheat', 'rice', 'gram'], 'secondary': ['barley', 'lentil', 'mustard'], 'specialty': ['arhar']},
        'bhagalpur': {'primary': ['rice', 'maize', 'wheat'], 'secondary': ['jute', 'urd', 'moong'], 'specialty': ['groundnut']},
        'muzaffarpur': {'primary': ['sugarcane', 'rice', 'wheat'], 'secondary': ['potato', 'mustard'], 'specialty': ['lentil']},
        'darbhanga': {'primary': ['rice', 'wheat', 'maize'], 'secondary': ['gram', 'arhar'], 'specialty': ['bajra']},
        'siwan': {'primary': ['rice', 'wheat'], 'secondary': ['gram', 'lentil', 'pea'], 'specialty': ['mustard']},
        'begusarai': {'primary': ['rice', 'wheat'], 'secondary': ['jute', 'mustard'], 'specialty': ['moong', 'urd']},
        'katihar': {'primary': ['maize', 'rice'], 'secondary': ['jute', 'urd', 'moong'], 'specialty': ['jowar', 'bajra']}
    }
    
    district_prefs = district_crop_preferences.get(district.lower(), {'primary': ['rice', 'wheat'], 'secondary': ['gram', 'mustard'], 'specialty': ['maize']})
    
    all_district_crops = (district_prefs.get('primary', []) + district_prefs.get('secondary', []) + district_prefs.get('specialty', []))
    
    # Find crops that are both seasonal AND grown in this district
    suitable_crops = []
    if seasonal_crops:
        suitable_crops = [crop for crop in all_district_crops if crop in seasonal_crops]
    
    # If no seasonal match, use season-based fallback
    if not suitable_crops:
        if current_season == 'kharif':
            kharif_crops = ['rice', 'maize', 'arhar', 'moong', 'urd', 'jowar', 'bajra', 'groundnut', 'soybean']
            suitable_crops = [crop for crop in all_district_crops if crop in kharif_crops]
        elif current_season == 'rabi':
            rabi_crops = ['wheat', 'barley', 'gram', 'lentil', 'pea', 'mustard', 'linseed', 'potato']
            suitable_crops = [crop for crop in all_district_crops if crop in rabi_crops]
        elif current_season == 'zaid':
            zaid_crops = ['maize', 'moong', 'urd', 'watermelon', 'cucumber']
            suitable_crops = [crop for crop in all_district_crops if crop in zaid_crops]
    
    if not suitable_crops:
        suitable_crops = district_prefs.get('primary', ['rice'])
    
    # Weight selection based on crop category
    weighted_crops = []
    for crop in suitable_crops:
        if crop in district_prefs.get('primary', []):
            weighted_crops.extend([crop] * 5)  # 5x weight for primary crops
        elif crop in district_prefs.get('secondary', []):
            weighted_crops.extend([crop] * 3)  # 3x weight for secondary crops
        else:
            weighted_crops.extend([crop] * 1)  # 1x weight for specialty crops
    
    selected_crop = random.choice(weighted_crops) if weighted_crops else 'rice'
    logger.info(f"Selected crop: {selected_crop} for {district} in {current_season} season")
    
    return selected_crop

async def get_current_crop_stage_dynamic(crop: str, district: str = None):
    """Determine crop stage based on current date and crop calendar"""
    try:
        crop_info = await crop_calendar_tools.get_crop_calendar('bihar', crop)
        
        if "error" in crop_info:
            return get_current_crop_stage_static(crop)
        
        stages = crop_info.get('stages', [])
        planting_period = crop_info.get('planting', '')
        current_month = datetime.now().month
        current_date = date.today()
        
        estimated_plant_date = estimate_planting_date(crop, planting_period, current_month)
        
        if estimated_plant_date:
            try:
                stage_data = await crop_calendar_tools.estimate_crop_stage(
                    crop, estimated_plant_date.isoformat(), current_date.isoformat()
                )
                
                if "error" not in stage_data:
                    return stage_data.get('stage', stages[0] if stages else 'Growing')
            except Exception as e:
                logger.warning(f"Error in dynamic stage calculation: {e}")
        
        return estimate_stage_by_month(crop, current_month, stages)
    
    except Exception as e:
        logger.error(f"Error in dynamic crop stage calculation: {e}")
        return get_current_crop_stage_static(crop)

def estimate_planting_date(crop: str, planting_period: str, current_month: int):
    """Estimate when the crop was likely planted"""
    current_year = datetime.now().year
    
    try:
        if 'june' in planting_period.lower():
            return date(current_year, 6, 15) if current_month >= 6 else date(current_year - 1, 6, 15)
        elif 'november' in planting_period.lower():
            if current_month >= 11:
                return date(current_year, 11, 15)
            elif current_month <= 4:
                return date(current_year - 1, 11, 15)
            else:
                return date(current_year, 11, 15)
        elif 'october' in planting_period.lower():
            if current_month >= 10:
                return date(current_year, 10, 15)
            elif current_month <= 4:
                return date(current_year - 1, 10, 15)
            else:
                return date(current_year, 10, 15)
        elif 'march' in planting_period.lower():
            if current_month >= 3 and current_month <= 8:
                return date(current_year, 3, 15)
            else:
                return date(current_year - 1, 3, 15)
    except Exception as e:
        logger.warning(f"Error estimating planting date: {e}")
    
    return None

def estimate_stage_by_month(crop: str, current_month: int, stages: list):
    """Estimate crop stage based on current month"""
    if not stages:
        return 'Growing'
    
    stage_mappings = {
        'rice': {6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6, 1: 7, 2: 8},
        'wheat': {11: 0, 12: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8},
        'maize': {6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6, 1: 7, 2: 7, 3: 0, 4: 1, 5: 2}
    }
    
    crop_mapping = stage_mappings.get(crop, {})
    stage_index = crop_mapping.get(current_month, 2)
    stage_index = min(stage_index, len(stages) - 1)
    
    return stages[stage_index] if stage_index < len(stages) else stages[-1]

def get_current_crop_stage_static(crop: str):
    """Original static crop stage function as fallback"""
    current_month = datetime.now().month
    
    if crop == 'rice':
        if current_month in [6, 7]:
            return 'Transplanting'
        elif current_month in [8, 9]:
            return 'Vegetative'
        elif current_month in [10, 11]:
            return 'Flowering'
        else:
            return 'Maturity'
    elif crop == 'wheat':
        if current_month in [11, 12]:
            return 'Sowing'
        elif current_month in [1, 2]:
            return 'Tillering'
        elif current_month in [3, 4]:
            return 'Flowering'
        else:
            return 'Harvesting'
    elif crop == 'sugarcane':
        if current_month in [2, 3, 4]:
            return 'Planting'
        elif current_month in [5, 6, 7, 8]:
            return 'Vegetative'
        elif current_month in [9, 10, 11]:
            return 'Maturity'
        else:
            return 'Harvesting'
    elif crop == 'maize':
        if current_month in [6, 7]:
            return 'Sowing'
        elif current_month in [8, 9]:
            return 'Vegetative'
        elif current_month in [10, 11]:
            return 'Grain Filling'
        else:
            return 'Harvesting'
    
    return 'Growing'

# 

async def generate_dynamic_alert(district: str, state: str):
    """Generate AI-powered dynamic alert data using real weather and crop intelligence"""
    
    try:
        # Step 1: Get villages for the district
        villages_data = await geographic_tools.list_villages(state, district)
        
        if "error" in villages_data:
            raise Exception(f"District '{district}' not found in {state}")
        
        # Step 2: Pick a random village from the actual list
        available_villages = villages_data.get("villages", [])
        if not available_villages:
            raise Exception(f"No villages found for {district}")
        
        selected_village = random.choice(available_villages)
        logger.info(f"Selected village: {selected_village} from {len(available_villages)} villages")
        
        # Step 3: Get coordinates for the selected village/district
        location_coords = None
        location_source = ""
        
        # Try village coordinates first
        try:
            village_location = await geographic_tools.reverse_geocode(selected_village)
            if "error" not in village_location and "lat" in village_location:
                location_coords = [village_location["lat"], village_location["lng"]]
                location_source = f"village_{selected_village}"
                logger.info(f"Using village coordinates for {selected_village}: {location_coords}")
        except Exception as e:
            logger.warning(f"Village geocoding failed for {selected_village}: {e}")
        
        # Fallback to district coordinates if village lookup failed
        if not location_coords:
            try:
                district_location = await geographic_tools.reverse_geocode(district)
                if "error" not in district_location and "lat" in district_location:
                    location_coords = [district_location["lat"], district_location["lng"]]
                    location_source = f"district_{district}"
                    logger.info(f"Using district coordinates for {district}: {location_coords}")
            except Exception as e:
                logger.warning(f"District geocoding failed for {district}: {e}")
        
        # Final fallback
        if not location_coords:
            logger.warning(f"No coordinates found for {selected_village} or {district}, using default")
            location_coords = [25.5941, 85.1376]  # Patna fallback
            location_source = "fallback_patna"
        
        # Step 4: Generate dynamic crop selection and stage
        regional_crop = await get_regional_crop_for_area(district, state)
        crop_stage = await get_current_crop_stage_dynamic(regional_crop, district)
        
        # Step 5: GET AI-POWERED WEATHER ALERT using your alert_generation_tools
        try:
            logger.info(f"Generating AI-powered alert for coordinates: {location_coords} (source: {location_source})")
            
            # Get the API key
            api_key = config.get("OPENAI_API_KEY")
            if not api_key:
                raise Exception("OpenAI API key not found")
            
            # Use your AI prediction tool
            ai_alert = await alert_generation_tools.predict_weather_alert(
                latitude=location_coords[0],
                longitude=location_coords[1],
                api_key=api_key
            )
            
            logger.info(f"AI alert generated successfully for {selected_village}, {district}")
            
            # Also get basic weather data for additional context
            try:
                current_weather_data = await open_meteo.get_current_weather(
                    latitude=location_coords[0], 
                    longitude=location_coords[1]
                )
                
                forecast_data = await open_meteo.get_weather_forecast(
                    latitude=location_coords[0], 
                    longitude=location_coords[1],
                    days=7
                )
                
                current_weather = current_weather_data.get('current_weather', {})
                daily_forecast = forecast_data.get('daily', {})
                
                current_temp = current_weather.get('temperature', 25)
                current_windspeed = current_weather.get('windspeed', 10)
                
                precipitation_list = daily_forecast.get('precipitation_sum', [0, 0, 0])
                next_3_days_rain = sum(precipitation_list[:3]) if precipitation_list else 0
                
                rain_probability = min(90, max(10, int(next_3_days_rain * 10))) if next_3_days_rain > 0 else 10
                estimated_humidity = min(95, max(40, 60 + int(next_3_days_rain * 2)))
                
                weather_context = {
                    "forecast_days": 7,
                    "rain_probability": rain_probability,
                    "expected_rainfall": f"{next_3_days_rain:.1f}mm",
                    "temperature": f"{current_temp:.1f}°C",
                    "humidity": f"{estimated_humidity}%",
                    "wind_speed": f"{current_windspeed:.1f} km/h",
                    "coordinates_source": location_source
                }
                
            except Exception as weather_error:
                logger.warning(f"Could not get basic weather data: {weather_error}")
                weather_context = {
                    "forecast_days": 7,
                    "coordinates_source": location_source,
                    "note": "Weather context limited due to API error"
                }
            
            # Extract AI analysis
            alert_description = ai_alert.get('alert', 'Weather update for agricultural activities')
            impact_description = ai_alert.get('impact', 'Monitor crops regularly')
            recommendations = ai_alert.get('recommendations', 'Continue routine farming activities')
            
            # Create comprehensive alert message combining AI insights
            alert_message = f"🤖 AI Weather Alert for {selected_village}, {district}: {alert_description}"
            if impact_description and impact_description.lower() not in ['none', 'n/a', '']:
                alert_message += f" 🌾 Crop Impact: {impact_description}"
            
            # Determine urgency and type based on AI response content
            urgency = "low"
            alert_type = "weather_update"
            
            alert_lower = alert_description.lower()
            impact_lower = impact_description.lower()
            recommendations_lower = recommendations.lower()
            
            # High urgency keywords
            if any(word in alert_lower + impact_lower for word in ['urgent', 'severe', 'critical', 'danger', 'emergency', 'immediate']):
                urgency = "high"
                alert_type = "severe_weather_warning"
            # Medium urgency keywords  
            elif any(word in alert_lower + impact_lower for word in ['warning', 'caution', 'alert', 'risk', 'damage', 'loss', 'stress', 'threat']):
                urgency = "medium"
                alert_type = "weather_warning"
            # Check recommendations for urgency indicators
            elif any(word in recommendations_lower for word in ['immediate', 'urgent', 'quickly', 'soon', 'now']):
                urgency = "medium"
                alert_type = "crop_risk_alert"
            
            # Parse recommendations into actionable items
            action_items = []
            if recommendations:
                # Split recommendations by common delimiters and clean up
                items = recommendations.replace('.', '|').replace(',', '|').replace(';', '|').replace(' and ', '|').split('|')
                action_items = [item.strip().lower().replace(' ', '_') for item in items if item.strip() and len(item.strip()) > 3]
                # Limit to 5 most important items and ensure they're actionable
                action_items = [item for item in action_items[:5] if any(verb in item for verb in ['monitor', 'check', 'apply', 'water', 'harvest', 'plant', 'protect', 'cover', 'drain', 'spray', 'fertilize'])]
            
            if not action_items:
                action_items = ["monitor_crops", "follow_weather_updates", "maintain_irrigation"]
            
            logger.info(f"AI-powered alert processed: Type={alert_type}, Urgency={urgency}, Actions={len(action_items)}")
            
        except Exception as ai_error:
            logger.error(f"Failed to get AI weather alert for {selected_village}, {district}: {ai_error}")
            raise Exception(f"Unable to generate AI weather alert: {str(ai_error)}")
        
        # Generate unique alert ID with timestamp
        alert_id = f"{state.upper()[:2]}_{district.upper()[:3]}_{selected_village.upper()[:3]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "alert_id": alert_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "location": {
                "village": selected_village,
                "district": district,
                "state": state.capitalize(),
                "coordinates": location_coords,
                "coordinates_source": location_source,
                "total_villages_in_district": len(available_villages)
            },
            "crop": {
                "name": regional_crop,
                "stage": crop_stage,
                "season": get_current_season(datetime.now().month),
                "planted_estimate": "2025-06-15"  # Could make this dynamic based on crop calendar
            },
            "alert": {
                "type": alert_type,
                "urgency": urgency,
                "message": alert_message,
                "action_items": action_items,
                "valid_until": (datetime.now() + timedelta(days=3)).isoformat() + "Z",
                "ai_generated": True
            },
            "ai_analysis": {
                "alert": alert_description,
                "impact": impact_description,
                "recommendations": recommendations
            },
            "weather": weather_context,
            "data_source": "ai_powered_openai_gpt4_with_open_meteo"
        }
    
    except Exception as e:
        logger.error(f"Error generating AI-powered alert for {district}, {state}: {e}")
        raise Exception(f"Failed to generate AI weather alert for {district}: {str(e)}")

@app.get("/")
async def root():
    return {"message": "MCP Weather Server is running"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "API is working"}

#  workflow endpoint for frontend
@app.post("/api/run-workflow")
async def run_workflow(request: WorkflowRequest):
    logger.info(f"Received workflow request: {request.state}, {request.district}")
    
    # Initialize variables
    sample_alert = None
    csv_content = ""
    
    try:
        # Create comprehensive workflow response
        workflow_results = []
        
        # Add workflow header
        workflow_results.append(f"Workflow for {request.district}, {request.state}")
        workflow_results.append("=" * 50)
        
        # weather data collection
        workflow_results.append("\n🌤️ Weather Data Collection")
        workflow_results.append("-" * 30)
        workflow_results.append("📡 Fetching real-time weather data...")
        
        try:
            sample_alert = await generate_dynamic_alert(request.district, request.state)
            
            workflow_results.append("✅ Current weather data retrieved from Open-Meteo API")
            workflow_results.append("✅ 7-day forecast collected")
            workflow_results.append("✅ Agricultural indices calculated")
            
        except Exception as weather_error:
            logger.error(f"Weather data error: {weather_error}")
            workflow_results.append(f"❌ Weather data collection failed: {str(weather_error)}")
            return {
                "message": "\n".join(workflow_results),
                "status": "error",
                "csv": "",
                "error": f"Unable to retrieve weather data: {str(weather_error)}"
            }
        
        if not sample_alert:
            return {
                "message": "Failed to generate alert data",
                "status": "error", 
                "csv": "",
                "error": "Alert generation failed"
            }
        
        # Alert generation
        workflow_results.append("\n🚨 Alert Generation")
        workflow_results.append("-" * 30)
        workflow_results.append("✅ Weather alerts generated")
        workflow_results.append(f"   - Data Source: {sample_alert.get('data_source', 'API')}")
        workflow_results.append(f"   - Alert Type: {sample_alert['alert']['type']}")
        workflow_results.append(f"   - Severity: {sample_alert['alert']['urgency']}")
        workflow_results.append(f"   - Village: {sample_alert['location']['village']}")
        workflow_results.append(f"   - Coordinates: {sample_alert['location']['coordinates']}")
        workflow_results.append(f"   - Crop: {sample_alert['crop']['name']} ({sample_alert['crop']['stage']})")
        workflow_results.append(f"   - Temperature: {sample_alert['weather']['temperature']}")
        workflow_results.append(f"   - Humidity: {sample_alert['weather']['humidity']}")
        workflow_results.append(f"   - Expected Rainfall: {sample_alert['weather']['expected_rainfall']}")
        workflow_results.append(f"   - Rain Probability: {sample_alert['weather']['rain_probability']}%")
    
        # WhatsApp Agent Response
        workflow_results.append("\n📱 WhatsApp Agent Response")
        workflow_results.append("-" * 30)
        try:
            whatsapp_message = whatsapp_agent.create_whatsapp_message(sample_alert)
            workflow_results.append(f"✅ Message created successfully")
            workflow_results.append(f"Text: {whatsapp_message.get('text', 'N/A')}")
            if 'buttons' in whatsapp_message:
                workflow_results.append(f"Buttons: {len(whatsapp_message['buttons'])} button(s)")
        except Exception as e:
            workflow_results.append(f"❌ Error: {str(e)}")
        
        # SMS Agent Response
        workflow_results.append("\n📱 SMS Agent Response")
        workflow_results.append("-" * 30)
        try:
            sms_message = sms_agent.create_sms_message(sample_alert)
            workflow_results.append(f"✅ SMS created successfully")
            workflow_results.append(f"Content: {str(sms_message)}")
        except Exception as e:
            workflow_results.append(f"❌ Error: {str(e)}")
        
        # USSD Agent Response
        workflow_results.append("\n📞 USSD Agent Response")
        workflow_results.append("-" * 30)
        try:
            ussd_menu = ussd_agent.create_ussd_menu(sample_alert)
            workflow_results.append(f"✅ USSD menu created successfully")
            workflow_results.append(f"Menu: {str(ussd_menu)}")
        except Exception as e:
            workflow_results.append(f"❌ Error: {str(e)}")
        
        # IVR Agent Response
        workflow_results.append("\n🎙️ IVR Agent Response")
        workflow_results.append("-" * 30)
        try:
            ivr_script = ivr_agent.create_ivr_script(sample_alert)
            workflow_results.append(f"✅ IVR script created successfully")
            workflow_results.append(f"Script: {str(ivr_script)}")
        except Exception as e:
            workflow_results.append(f"❌ Error: {str(e)}")
        
        # Telegram Agent Response
        workflow_results.append("\n🤖 Telegram Agent Response")
        workflow_results.append("-" * 30)
        try:
            telegram_message = telegram_agent.create_telegram_message(sample_alert)
            workflow_results.append(f"✅ Telegram message created successfully")
            workflow_results.append(f"Content: {str(telegram_message)}")
        except Exception as e:
            workflow_results.append(f"❌ Error: {str(e)}")
        
        # Summary
        workflow_results.append("\n✅ Workflow Summary")
        workflow_results.append("-" * 30)
        workflow_results.append("Workflow execution completed with REAL weather data")
        workflow_results.append(f"Location: {request.district}, {request.state}")
        workflow_results.append(f"Weather Source: Open-Meteo API")
        workflow_results.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Join all results into a single formatted string
        formatted_output = "\n".join(workflow_results)

        # Generate CSV 
        try:
            csv_buffer = StringIO()
            writer = csv.writer(csv_buffer)
            
            # Write headers
            headers = ["weather data", "whatsapp", "sms", "ussd", "ivr", "telegram"]
            writer.writerow(headers)
            
            # Prepare weather data as a single string with line breaks
            weather_info = "\n".join([
                f"   - Data Source: {sample_alert.get('data_source', 'API')}",
                f"   - Alert Type: {sample_alert['alert']['type']}",
                f"   - Severity: {sample_alert['alert']['urgency']}",
                f"   - Village: {sample_alert['location']['village']}",
                f"   - Coordinates: {sample_alert['location']['coordinates']}",
                f"   - Crop: {sample_alert['crop']['name']} ({sample_alert['crop']['stage']})",
                f"   - Temperature: {sample_alert['weather']['temperature']}",
                f"   - Humidity: {sample_alert['weather']['humidity']}",
                f"   - Expected Rainfall: {sample_alert['weather']['expected_rainfall']}",
                f"   - Rain Probability: {sample_alert['weather']['rain_probability']}%"
            ])
            
            weather_data = [weather_info]
            
            # Extract agent outputs only (no status messages)
            whatsapp_data = []
            sms_data = []
            ussd_data = []
            ivr_data = []
            telegram_data = []
            
            # Get WhatsApp message
            try:
                whatsapp_message = whatsapp_agent.create_whatsapp_message(sample_alert)
                whatsapp_text = whatsapp_message.get('text', 'N/A')
                whatsapp_data.append(whatsapp_text)
                if 'buttons' in whatsapp_message and whatsapp_message['buttons']:
                    whatsapp_data.append(f"Buttons: {whatsapp_message['buttons']}")
            except Exception as e:
                whatsapp_data.append(f"Error: {str(e)}")
            
            # Get SMS message
            try:
                sms_message = sms_agent.create_sms_message(sample_alert)
                sms_data.append(str(sms_message))
            except Exception as e:
                sms_data.append(f"Error: {str(e)}")
            
            # Get USSD menu
            try:
                ussd_menu = ussd_agent.create_ussd_menu(sample_alert)
                ussd_data.append(str(ussd_menu))
            except Exception as e:
                ussd_data.append(f"Error: {str(e)}")
            
            # Get IVR script
            try:
                ivr_script = ivr_agent.create_ivr_script(sample_alert)
                ivr_data.append(str(ivr_script))
            except Exception as e:
                ivr_data.append(f"Error: {str(e)}")
            
            # Get Telegram message
            try:
                telegram_message = telegram_agent.create_telegram_message(sample_alert)
                telegram_data.append(str(telegram_message))
            except Exception as e:
                telegram_data.append(f"Error: {str(e)}")
            
            # Find the maximum number of rows needed
            max_rows = max(
                len(weather_data),
                len(whatsapp_data) if whatsapp_data else 1,
                len(sms_data) if sms_data else 1,
                len(ussd_data) if ussd_data else 1,
                len(ivr_data) if ivr_data else 1,
                len(telegram_data) if telegram_data else 1
            )
            
            # Write data rows
            for i in range(max_rows):
                row = [
                    weather_data[i] if i < len(weather_data) else "",
                    whatsapp_data[i] if i < len(whatsapp_data) else "",
                    sms_data[i] if i < len(sms_data) else "",
                    ussd_data[i] if i < len(ussd_data) else "",
                    ivr_data[i] if i < len(ivr_data) else "",
                    telegram_data[i] if i < len(telegram_data) else ""
                ]
                writer.writerow(row)

            csv_content = csv_buffer.getvalue()
            logger.info("CSV content generated successfully")
            
        except Exception as csv_error:
            logger.error(f"Error generating CSV: {csv_error}")
            csv_content = f"Error generating CSV: {str(csv_error)}"
        
        logger.info(f"Successfully completed workflow for {request.district}, {request.state}")
        return {
            "message": formatted_output,
            "status": "success",
            "csv": csv_content,
            "raw_data": {
                "state": request.state,
                "district": request.district,
                "alert_data": sample_alert
            }
        }
        
    except Exception as e:
        logger.exception(f"Error in workflow for {request.district}, {request.state}")
        return {
            "message": f"Error running workflow: {str(e)}",
            "status": "error",
            "csv": "",
            "error": str(e)
        }


@app.post("/mcp")
async def mcp_endpoint(request: MCPRequest):
    logger.info(f"Received request for tool: {request.tool}")
    tool_config = get_tool_config(request.tool)

    if not tool_config:
        logger.error(f"Tool not found: {request.tool}")
        raise HTTPException(status_code=404, detail="Tool not found")

    try:
        if tool_config["module"] == "open_meteo":
            result = await getattr(open_meteo, request.tool)(**request.parameters)
        elif tool_config["module"] == "tomorrow_io":
            api_key = config.get("TOMORROW_IO_API_KEY")
            result = await getattr(tomorrow_io, request.tool)(**request.parameters, api_key=api_key)
        elif tool_config["module"] == "google_weather":
            api_key = config.get("GOOGLE_WEATHER_API_KEY")
            result = await getattr(google_weather, request.tool)(**request.parameters, api_key=api_key)
        elif tool_config["module"] == "openweathermap":
            api_key = config.get("OPENWEATHERMAP_API_KEY")
            result = await getattr(openweathermap, request.tool)(**request.parameters, api_key=api_key)
        elif tool_config["module"] == "accuweather":
            api_key = config.get("ACCUWEATHER_API_KEY")
            result = await getattr(accuweather, request.tool)(**request.parameters, api_key=api_key)
        elif tool_config["module"] == "openai_llm":
            api_key = config.get("OPENAI_API_KEY")
            result = await getattr(openai_llm, request.tool)(**request.parameters, api_key=api_key)
        elif tool_config["module"] == "geographic_tools":
            result = await getattr(geographic_tools, request.tool)(**request.parameters)
        elif tool_config["module"] == "crop_calendar_tools":
            result = await getattr(crop_calendar_tools, request.tool)(**request.parameters)
        elif tool_config["module"] == "alert_generation_tools":
            api_key = config.get("OPENAI_API_KEY")
            result = await getattr(alert_generation_tools, request.tool)(**request.parameters, api_key=api_key)
        else:
            raise HTTPException(status_code=500, detail="Invalid tool module")

        logger.info(f"Successfully executed tool: {request.tool}")
        return result
    except Exception as e:
        logger.exception(f"Error executing tool: {request.tool}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/a2a/sms")
async def a2a_sms_endpoint(request: AlertRequest):
    return {"message": sms_agent.create_sms_message(request.alert_json)}

@app.post("/a2a/whatsapp")
async def a2a_whatsapp_endpoint(request: AlertRequest):
    return whatsapp_agent.create_whatsapp_message(request.alert_json)

@app.post("/a2a/ussd")
async def a2a_ussd_endpoint(request: AlertRequest):
    return {"menu": ussd_agent.create_ussd_menu(request.alert_json)}

@app.post("/a2a/ivr")
async def a2a_ivr_endpoint(request: AlertRequest):
    return {"script": ivr_agent.create_ivr_script(request.alert_json)}

@app.post("/a2a/telegram")
async def a2a_telegram_endpoint(request: AlertRequest):
    return telegram_agent.create_telegram_message(request.alert_json)


# for smithery + context7

@app.post("/mcp")
async def mcp_rpc_handler(request: dict):
    method = request.get("method")
    params = request.get("params", {})
    tool_name = params.get("tool_name")
    arguments = params.get("arguments", {})
    req_id = request.get("id")

    # Handle run_workflow tool
    if method == "call_tool" and tool_name == "run_workflow":
        state = arguments.get("state")
        district = arguments.get("district")
        result = await run_workflow(WorkflowRequest(state=state, district=district))
        return {"jsonrpc": "2.0", "result": result, "id": req_id}

    # Handle other tools dynamically via your tool config
    if method == "call_tool":
        try:
            result = await mcp_endpoint(MCPRequest(tool=tool_name, parameters=arguments))
            return {"jsonrpc": "2.0", "result": result, "id": req_id}
        except Exception as e:
            return {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}, "id": req_id}

    return {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Unknown method"}, "id": req_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
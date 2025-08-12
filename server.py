import os
import logging
import random
from datetime import datetime, timedelta, date
from typing import Optional, Dict, List, Any
import csv
from io import StringIO

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import dotenv_values

# Import tools
from tools import (
    open_meteo, 
    tomorrow_io, 
    google_weather, 
    openweathermap, 
    accuweather, 
    openai_llm, 
    geographic_tools, 
    crop_calendar_tools, 
    alert_generation_tools
)
from a2a_agents import sms_agent, whatsapp_agent, ussd_agent, ivr_agent, telegram_agent
from utils.weather_utils import get_tool_config

# Configuration
config = dotenv_values(".env")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Verify API keys
openai_key = config.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_key:
    logger.warning("OpenAI API key not found - AI features will be limited")
else:
    logger.info("OpenAI API key found")

app = FastAPI(title="MCP Weather Server", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mcp-ui.vercel.app", "*"],  # Add * for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Pydantic models
class MCPRequest(BaseModel):
    tool: str
    parameters: dict

class AlertRequest(BaseModel):
    alert_json: dict

class WorkflowRequest(BaseModel):
    state: str
    district: str

# Crop calendar constants
CROP_CALENDAR = {
    "rice": {
        "season": "Kharif",
        "planting": "June-July",
        "harvesting": "October-November",
        "duration_days": 120,
        "stages": ["Nursery/Seedling", "Transplanting", "Vegetative", "Tillering",
                  "Panicle Initiation", "Flowering", "Milk/Dough", "Maturity", "Harvesting"]
    },
    "wheat": {
        "season": "Rabi",
        "planting": "November-December",
        "harvesting": "March-April",
        "duration_days": 120,
        "stages": ["Sowing", "Germination", "Tillering", "Jointing", "Booting",
                  "Heading", "Flowering", "Grain Filling", "Maturity", "Harvesting"]
    },
    "maize": {
        "season": "Kharif/Zaid",
        "planting": "June-July / March-April",
        "harvesting": "September-October / June",
        "duration_days": 110,
        "stages": ["Sowing", "Emergence", "Vegetative", "Tasseling", "Silking",
                  "Grain Filling", "Maturity", "Harvesting"]
    },
    "sugarcane": {
        "season": "Annual",
        "planting": "February-March",
        "harvesting": "December-January",
        "duration_days": 300,
        "stages": ["Planting", "Germination", "Tillering", "Grand Growth",
                  "Maturation", "Ripening", "Harvesting"]
    },
    "mustard": {
        "season": "Rabi",
        "planting": "October-November",
        "harvesting": "February-March",
        "duration_days": 110,
        "stages": ["Sowing", "Germination", "Rosette", "Stem Elongation", 
                  "Flowering", "Pod Formation", "Pod Filling", "Maturity", "Harvesting"]
    }
}

# District-specific crop preferences for Bihar
DISTRICT_CROPS = {
    'patna': {'primary': ['rice', 'wheat', 'potato'], 'secondary': ['mustard', 'gram'], 'specialty': ['sugarcane']},
    'gaya': {'primary': ['wheat', 'rice'], 'secondary': ['barley', 'mustard'], 'specialty': ['gram']},
    'bhagalpur': {'primary': ['rice', 'maize', 'wheat'], 'secondary': ['jute'], 'specialty': ['groundnut']},
    'muzaffarpur': {'primary': ['sugarcane', 'rice', 'wheat'], 'secondary': ['potato', 'mustard'], 'specialty': ['lentil']},
    'darbhanga': {'primary': ['rice', 'wheat', 'maize'], 'secondary': ['gram'], 'specialty': ['bajra']},
    'siwan': {'primary': ['rice', 'wheat'], 'secondary': ['gram', 'lentil'], 'specialty': ['mustard']},
    'begusarai': {'primary': ['rice', 'wheat'], 'secondary': ['jute', 'mustard'], 'specialty': ['moong']},
    'katihar': {'primary': ['maize', 'rice'], 'secondary': ['jute'], 'specialty': ['jowar']}
}

def get_current_season(month: int) -> str:
    """Determine current agricultural season"""
    if month in [6, 7, 8, 9]:  # June to September
        return 'kharif'
    elif month in [10, 11, 12, 1, 2, 3]:  # October to March
        return 'rabi'
    else:  # April, May
        return 'zaid'

def select_regional_crop(district: str, state: str) -> str:
    """Select appropriate crop based on district, season, and preferences"""
    if state.lower() != 'bihar':
        return 'rice'  # fallback
    
    current_month = datetime.now().month
    current_season = get_current_season(current_month)
    
    # Get district preferences
    district_prefs = DISTRICT_CROPS.get(district.lower(), {
        'primary': ['rice', 'wheat'], 
        'secondary': ['gram'], 
        'specialty': ['maize']
    })
    
    # Season-specific crop filtering
    seasonal_crops = {
        'kharif': ['rice', 'maize', 'sugarcane', 'jowar', 'bajra', 'groundnut'],
        'rabi': ['wheat', 'barley', 'gram', 'lentil', 'mustard', 'potato'],
        'zaid': ['maize', 'moong', 'watermelon', 'cucumber']
    }
    
    # Combine district and seasonal preferences
    all_district_crops = (district_prefs.get('primary', []) + 
                         district_prefs.get('secondary', []) + 
                         district_prefs.get('specialty', []))
    
    suitable_crops = [crop for crop in all_district_crops 
                     if crop in seasonal_crops.get(current_season, [])]
    
    if not suitable_crops:
        suitable_crops = district_prefs.get('primary', ['rice'])
    
    # Weighted selection (primary crops more likely)
    weighted_crops = []
    for crop in suitable_crops:
        if crop in district_prefs.get('primary', []):
            weighted_crops.extend([crop] * 5)
        elif crop in district_prefs.get('secondary', []):
            weighted_crops.extend([crop] * 3)
        else:
            weighted_crops.extend([crop] * 1)
    
    selected_crop = random.choice(weighted_crops) if weighted_crops else 'rice'
    logger.info(f"Selected crop: {selected_crop} for {district} in {current_season} season")
    
    return selected_crop

def estimate_crop_stage(crop: str, current_month: int) -> str:
    """Estimate current crop stage based on crop type and month"""
    if crop not in CROP_CALENDAR:
        return 'Growing'
    
    crop_data = CROP_CALENDAR[crop]
    stages = crop_data['stages']
    
    # Month-based stage estimation
    stage_mappings = {
        'rice': {6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6, 1: 7, 2: 8},
        'wheat': {11: 0, 12: 1, 1: 2, 2: 3, 3: 4, 4: 5},
        'maize': {6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 3: 0, 4: 1, 5: 2},
        'sugarcane': {2: 0, 3: 1, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 6, 12: 6, 1: 6},
        'mustard': {10: 0, 11: 1, 12: 2, 1: 3, 2: 4, 3: 5}
    }
    
    crop_mapping = stage_mappings.get(crop, {})
    stage_index = crop_mapping.get(current_month, len(stages) // 2)  # Default to middle stage
    stage_index = min(stage_index, len(stages) - 1)
    
    return stages[stage_index] if stages else 'Growing'

async def get_location_coordinates(village: str, district: str) -> tuple[list, str]:
    """Get coordinates for village or district with fallback"""
    location_coords = None
    location_source = ""
    
    # Try village coordinates first
    try:
        village_location = await geographic_tools.reverse_geocode(village)
        if "error" not in village_location and "latitude" in village_location:
            location_coords = [village_location["latitude"], village_location["longitude"]]
            location_source = f"village_{village}"
            logger.info(f"Using village coordinates for {village}: {location_coords}")
    except Exception as e:
        logger.warning(f"Village geocoding failed for {village}: {e}")
    
    # Fallback to district coordinates
    if not location_coords:
        try:
            district_location = await geographic_tools.reverse_geocode(district)
            if "error" not in district_location and "latitude" in district_location:
                location_coords = [district_location["latitude"], district_location["longitude"]]
                location_source = f"district_{district}"
                logger.info(f"Using district coordinates for {district}: {location_coords}")
        except Exception as e:
            logger.warning(f"District geocoding failed for {district}: {e}")
    
    # Final fallback
    if not location_coords:
        logger.warning(f"No coordinates found, using Patna fallback")
        location_coords = [25.5941, 85.1376]  # Patna coordinates
        location_source = "fallback_patna"
    
    return location_coords, location_source

async def generate_weather_based_alert(weather_data: dict, crop: str, crop_stage: str, 
                                     village: str, district: str) -> tuple[str, str, str, list]:
    """Generate alert based on weather conditions"""
    current_weather = weather_data.get('current_weather', {})
    daily_forecast = weather_data.get('daily', {})
    
    current_temp = current_weather.get('temperature', 25)
    current_windspeed = current_weather.get('windspeed', 10)
    
    precipitation_list = daily_forecast.get('precipitation_sum', [0, 0, 0])
    next_3_days_rain = sum(precipitation_list[:3]) if precipitation_list else 0
    
    # Generate alert based on conditions
    if next_3_days_rain > 25:
        alert_type = "heavy_rain_warning"
        urgency = "high"
        alert_message = (f"Heavy rainfall ({next_3_days_rain:.1f}mm) expected in next 3 days "
                        f"near {village}, {district}. {crop} at {crop_stage} stage may be affected. "
                        f"Delay fertilizer application and ensure proper drainage.")
        action_items = ["delay_fertilizer", "check_drainage", "monitor_crops", "prepare_harvest_protection"]
    
    elif next_3_days_rain > 10:
        alert_type = "moderate_rain_warning"
        urgency = "medium"
        alert_message = (f"Moderate rainfall ({next_3_days_rain:.1f}mm) expected in next 3 days "
                        f"near {village}, {district}. Monitor {crop} at {crop_stage} stage carefully.")
        action_items = ["monitor_soil", "check_drainage", "adjust_irrigation"]
    
    elif next_3_days_rain < 2 and current_temp > 35:
        alert_type = "heat_drought_warning"
        urgency = "high"
        alert_message = (f"High temperature ({current_temp:.1f}°C) with minimal rainfall expected "
                        f"near {village}, {district}. {crop} at {crop_stage} stage needs extra care. "
                        f"Increase irrigation frequency.")
        action_items = ["increase_irrigation", "mulch_crops", "monitor_plant_stress"]
    
    elif current_temp < 10:
        alert_type = "cold_warning"
        urgency = "medium"
        alert_message = (f"Low temperature ({current_temp:.1f}°C) expected near {village}, {district}. "
                        f"Protect {crop} crops from cold damage.")
        action_items = ["protect_crops", "cover_seedlings", "adjust_irrigation_timing"]
    
    elif current_windspeed > 30:
        alert_type = "high_wind_warning"
        urgency = "medium"
        alert_message = (f"High winds ({current_windspeed:.1f} km/h) expected near {village}, {district}. "
                        f"Secure {crop} crop supports and structures.")
        action_items = ["secure_supports", "check_structures", "monitor_damage"]
    
    else:
        alert_type = "weather_update"
        urgency = "low"
        alert_message = (f"Normal weather conditions expected near {village}, {district}. "
                        f"{crop} at {crop_stage} stage. Temperature {current_temp:.1f}°C, "
                        f"rainfall {next_3_days_rain:.1f}mm.")
        action_items = ["routine_monitoring", "maintain_irrigation"]
    
    return alert_type, urgency, alert_message, action_items

async def generate_ai_alert(latitude: float, longitude: float, crop: str, 
                          crop_stage: str, village: str, district: str) -> Optional[dict]:
    """Generate AI-powered weather alert using available tools"""
    if not openai_key:
        logger.warning("No OpenAI API key - skipping AI alert generation")
        return None
    
    try:
        # First get weather data for the AI context
        current_weather_data = await open_meteo.get_current_weather(
            latitude=latitude, longitude=longitude
        )
        forecast_data = await open_meteo.get_weather_forecast(
            latitude=latitude, longitude=longitude, days=7
        )
        
        # Prepare weather context for AI
        current_weather = current_weather_data.get('current_weather', {})
        daily_forecast = forecast_data.get('daily', {})
        
        weather_context = {
            'temperature': current_weather.get('temperature', 25),
            'windspeed': current_weather.get('windspeed', 10),
            'precipitation_forecast': daily_forecast.get('precipitation_sum', [0, 0, 0])[:3]
        }
        
        ai_alert = await alert_generation_tools.generate_weather_alert(
            crop=crop,
            weather_data=weather_context,
            growth_stage=crop_stage,
            api_key=openai_key,
            latitude=latitude,
            longitude=longitude
        )
        
        # Extract AI response 
        if isinstance(ai_alert, dict):
            alert_description = ai_alert.get('alert', 'Weather update for agricultural activities')
            impact_description = ai_alert.get('impact', 'Monitor crops regularly') 
            recommendations = ai_alert.get('recommendations', 'Continue routine farming activities')
        else:
            # If it returns a string or other format
            alert_description = str(ai_alert)
            impact_description = 'Monitor crops regularly'
            recommendations = 'Follow weather updates and maintain good practices'
        
        # Create enhanced alert message
        alert_message = f"🤖 AI Weather Alert for {village}, {district}: {alert_description}"
        if impact_description and impact_description.lower() not in ['none', 'n/a', '']:
            alert_message += f" 🌾 Crop Impact ({crop} - {crop_stage}): {impact_description}"
        
        return {
            'alert': alert_description,
            'impact': impact_description,
            'recommendations': recommendations,
            'enhanced_message': alert_message
        }
        
    except Exception as e:
        logger.error(f"AI alert generation failed: {e}")
        return None

async def generate_dynamic_alert(district: str, state: str) -> dict:
    """Main function to generate comprehensive weather alert"""
    try:
        # Step 1: Get villages for the district
        villages_data = await geographic_tools.list_villages(state, district)
        
        if "error" in villages_data:
            raise Exception(f"District '{district}' not found in {state}")
        
        available_villages = villages_data.get("villages", [])
        if not available_villages:
            raise Exception(f"No villages found for {district}")
        
        # Step 2: Select random village
        selected_village = random.choice(available_villages)
        logger.info(f"Selected village: {selected_village} from {len(available_villages)} villages")
        
        # Step 3: Get coordinates
        location_coords, location_source = await get_location_coordinates(selected_village, district)
        
        # Step 4: Generate crop selection and stage
        regional_crop = select_regional_crop(district, state)
        crop_stage = estimate_crop_stage(regional_crop, datetime.now().month)
        
        # Step 5: Get weather data
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
            
            logger.info(f"Weather data retrieved for {selected_village}, {district}")
            
        except Exception as weather_error:
            logger.error(f"Failed to get weather data: {weather_error}")
            raise Exception(f"Unable to retrieve weather conditions for {selected_village}, {district}")
        
        # Step 6: Generate alerts
        alert_type, urgency, alert_message, action_items = await generate_weather_based_alert(
            {'current_weather': current_weather_data.get('current_weather', {}),
             'daily': forecast_data.get('daily', {})},
            regional_crop, crop_stage, selected_village, district
        )
        
        # Step 7: Try to enhance with AI if available
        ai_analysis = await generate_ai_alert(
            location_coords[0], location_coords[1], 
            regional_crop, crop_stage, selected_village, district
        )
        
        # Step 8: Prepare weather context
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
        
        # Generate unique alert ID
        alert_id = f"{state.upper()[:2]}_{district.upper()[:3]}_{selected_village.upper()[:3]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Build final response
        result = {
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
                "planted_estimate": "2025-06-15"
            },
            "alert": {
                "type": alert_type,
                "urgency": urgency,
                "message": ai_analysis['enhanced_message'] if ai_analysis else alert_message,
                "action_items": action_items,
                "valid_until": (datetime.now() + timedelta(days=3)).isoformat() + "Z",
                "ai_generated": ai_analysis is not None
            },
            "weather": weather_context,
            "data_source": "open_meteo_api_with_ai_enhancement" if ai_analysis else "open_meteo_api"
        }
        
        if ai_analysis:
            result["ai_analysis"] = {
                "alert": ai_analysis['alert'],
                "impact": ai_analysis['impact'],
                "recommendations": ai_analysis['recommendations']
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating alert for {district}, {state}: {e}")
        raise Exception(f"Failed to generate weather alert for {district}: {str(e)}")

# API Routes
@app.get("/")
async def root():
    return {"message": "MCP Weather Server v1.0 - AI-Powered Agricultural Alerts", "status": "running"}

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "API is working",
        "openai_available": openai_key is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/run-workflow")
async def run_workflow(request: WorkflowRequest):
    """Main workflow endpoint for generating comprehensive agricultural alerts"""
    logger.info(f"Received workflow request: {request.state}, {request.district}")
    
    workflow_results = []
    
    try:
        # Workflow header
        workflow_results.extend([
            f"🌾 Agricultural Alert Workflow for {request.district.title()}, {request.state.title()}",
            "=" * 70,
            "",
            "🌤️ Weather Data Collection",
            "-" * 30
        ])
        
        # Generate dynamic alert
        sample_alert = await generate_dynamic_alert(request.district, request.state)
        
        workflow_results.extend([
            "✅ Weather data retrieved successfully",
            f"   📍 Location: {sample_alert['location']['village']}, {sample_alert['location']['district']}",
            f"   🌡️ Temperature: {sample_alert['weather']['temperature']}",
            f"   🌧️ Expected Rainfall: {sample_alert['weather']['expected_rainfall']}",
            f"   💨 Wind Speed: {sample_alert['weather']['wind_speed']}",
            f"   🌾 Crop: {sample_alert['crop']['name']} ({sample_alert['crop']['stage']})",
            f"   🚨 Alert Type: {sample_alert['alert']['type']} - {sample_alert['alert']['urgency'].upper()} priority",
            ""
        ])
        
        # Generate agent responses
        agents = [
            ("📱 WhatsApp Agent", whatsapp_agent.create_whatsapp_message),
            ("📱 SMS Agent", sms_agent.create_sms_message),
            ("📞 USSD Agent", ussd_agent.create_ussd_menu),
            ("🎙️ IVR Agent", ivr_agent.create_ivr_script),
            ("🤖 Telegram Agent", telegram_agent.create_telegram_message)
        ]
        
        agent_responses = {}
        
        for agent_name, agent_func in agents:
            workflow_results.extend([agent_name, "-" * 30])
            try:
                response = agent_func(sample_alert)
                workflow_results.append("✅ Message generated successfully")
                agent_responses[agent_name] = response
                
                # Add preview for some agents
                if "WhatsApp" in agent_name:
                    text = response.get('text', 'N/A')
                    workflow_results.append(f"   Preview: {text[:100]}..." if len(text) > 100 else f"   Preview: {text}")
                elif "SMS" in agent_name:
                    workflow_results.append(f"   Preview: {str(response)[:100]}...")
                
            except Exception as e:
                workflow_results.append(f"❌ Error: {str(e)}")
                agent_responses[agent_name] = f"Error: {str(e)}"
            
            workflow_results.append("")
        
        # Summary
        workflow_results.extend([
            "✅ Workflow Summary",
            "-" * 30,
            f"🎯 Successfully generated alerts for {sample_alert['location']['village']}, {request.district}",
            f"📊 Data Sources: {sample_alert['data_source']}",
            f"🤖 AI Enhanced: {'Yes' if sample_alert['alert']['ai_generated'] else 'No'}",
            f"⏰ Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"📱 Agents Processed: {len([r for r in agent_responses.values() if not str(r).startswith('Error:')])}/{len(agents)}"
        ])
        
        # Generate CSV
        csv_content = generate_csv_export(sample_alert, agent_responses)
        
        return {
            "message": "\n".join(workflow_results),
            "status": "success",
            "csv": csv_content,
            "raw_data": {
                "state": request.state,
                "district": request.district,
                "alert_data": sample_alert,
                "agent_responses": agent_responses
            }
        }
        
    except Exception as e:
        error_msg = f"❌ Workflow failed: {str(e)}"
        workflow_results.append(error_msg)
        logger.exception(f"Workflow error for {request.district}, {request.state}")
        
        return {
            "message": "\n".join(workflow_results),
            "status": "error",
            "csv": "",
            "error": str(e)
        }

def generate_csv_export(alert_data: dict, agent_responses: dict) -> str:
    """Generate CSV export of workflow results"""
    try:
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        
        # Headers
        headers = ["Field", "Value"]
        writer.writerow(headers)
        
        # Alert data
        writer.writerow(["Alert ID", alert_data["alert_id"]])
        writer.writerow(["Village", alert_data["location"]["village"]])
        writer.writerow(["District", alert_data["location"]["district"]])
        writer.writerow(["State", alert_data["location"]["state"]])
        writer.writerow(["Coordinates", str(alert_data["location"]["coordinates"])])
        writer.writerow(["Crop", alert_data["crop"]["name"]])
        writer.writerow(["Crop Stage", alert_data["crop"]["stage"]])
        writer.writerow(["Temperature", alert_data["weather"]["temperature"]])
        writer.writerow(["Rainfall", alert_data["weather"]["expected_rainfall"]])
        writer.writerow(["Alert Type", alert_data["alert"]["type"]])
        writer.writerow(["Urgency", alert_data["alert"]["urgency"]])
        writer.writerow(["Alert Message", alert_data["alert"]["message"]])
        
        # Agent responses
        writer.writerow([])  # Empty row
        writer.writerow(["Agent", "Response"])
        for agent_name, response in agent_responses.items():
            clean_agent_name = agent_name.replace("📱 ", "").replace("📞 ", "").replace("🎙️ ", "").replace("🤖 ", "")
            writer.writerow([clean_agent_name, str(response)[:500]])  # Limit response length
        
        return csv_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"CSV generation error: {e}")
        return f"Error generating CSV: {str(e)}"

# Other API endpoints
@app.post("/mcp")
async def mcp_endpoint(request: MCPRequest):
    """MCP tool execution endpoint"""
    logger.info(f"Received request for tool: {request.tool}")
    tool_config = get_tool_config(request.tool)

    if not tool_config:
        logger.error(f"Tool not found: {request.tool}")
        raise HTTPException(status_code=404, detail="Tool not found")

    try:
        # Route to appropriate module
        module_map = {
            "open_meteo": open_meteo,
            "tomorrow_io": tomorrow_io,
            "google_weather": google_weather,
            "openweathermap": openweathermap,
            "accuweather": accuweather,
            "openai_llm": openai_llm,
            "geographic_tools": geographic_tools,
            "crop_calendar_tools": crop_calendar_tools,
            "alert_generation_tools": alert_generation_tools
        }
        
        module = module_map.get(tool_config["module"])
        if not module:
            raise HTTPException(status_code=500, detail="Invalid tool module")
        
        # Add API key if needed
        params = request.parameters.copy()
        if tool_config["module"] in ["tomorrow_io", "google_weather", "openweathermap", "accuweather"]:
            api_key_map = {
                "tomorrow_io": "TOMORROW_IO_API_KEY",
                "google_weather": "GOOGLE_WEATHER_API_KEY", 
                "openweathermap": "OPENWEATHERMAP_API_KEY",
                "accuweather": "ACCUWEATHER_API_KEY"
            }
            key_name = api_key_map[tool_config["module"]]
            params["api_key"] = config.get(key_name)
        elif tool_config["module"] in ["openai_llm", "alert_generation_tools"]:
            params["api_key"] = openai_key
        
        # Execute tool function
        result = await getattr(module, request.tool)(**params)
        
        logger.info(f"Successfully executed tool: {request.tool}")
        return result
        
    except Exception as e:
        logger.exception(f"Error executing tool: {request.tool}")
        raise HTTPException(status_code=500, detail=str(e))

# A2A Agent endpoints
@app.post("/a2a/sms")
async def a2a_sms_endpoint(request: AlertRequest):
    """SMS agent endpoint"""
    return {"message": sms_agent.create_sms_message(request.alert_json)}

@app.post("/a2a/whatsapp")
async def a2a_whatsapp_endpoint(request: AlertRequest):
    """WhatsApp agent endpoint"""
    return whatsapp_agent.create_whatsapp_message(request.alert_json)

@app.post("/a2a/ussd")
async def a2a_ussd_endpoint(request: AlertRequest):
    """USSD agent endpoint"""
    return {"menu": ussd_agent.create_ussd_menu(request.alert_json)}

@app.post("/a2a/ivr")
async def a2a_ivr_endpoint(request: AlertRequest):
    """IVR agent endpoint"""
    return {"script": ivr_agent.create_ivr_script(request.alert_json)}

@app.post("/a2a/telegram")
async def a2a_telegram_endpoint(request: AlertRequest):
    """Telegram agent endpoint"""
    return telegram_agent.create_telegram_message(request.alert_json)

# MCP RPC handler 
@app.post("/mcp-rpc")
async def mcp_rpc_handler(request: dict):
    """JSON-RPC handler for MCP integration"""
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

    # Handle other tools dynamically
    if method == "call_tool":
        try:
            result = await mcp_endpoint(MCPRequest(tool=tool_name, parameters=arguments))
            return {"jsonrpc": "2.0", "result": result, "id": req_id}
        except Exception as e:
            return {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}, "id": req_id}

    return {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Unknown method"}, "id": req_id}

# Additional utility endpoints
@app.get("/api/districts/{state}")
async def get_districts(state: str):
    """Get list of districts for a state"""
    try:
        result = await geographic_tools.list_villages(state)
        if "districts" in result:
            return {"districts": result["districts"]}
        return {"error": "State not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/villages/{state}/{district}")
async def get_villages(state: str, district: str):
    """Get list of villages for a district"""
    try:
        result = await geographic_tools.list_villages(state, district)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/crops/{region}")
async def get_crops(region: str):
    """Get list of crops for a region"""
    try:
        result = await crop_calendar_tools.get_crop_calendar(region)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/weather/{latitude}/{longitude}")
async def get_weather(latitude: float, longitude: float):
    """Get current weather for coordinates"""
    try:
        current_weather = await open_meteo.get_current_weather(latitude=latitude, longitude=longitude)
        forecast = await open_meteo.get_weather_forecast(latitude=latitude, longitude=longitude, days=7)
        
        return {
            "current": current_weather,
            "forecast": forecast,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception("Unhandled exception occurred")
    return {
        "error": "Internal server error",
        "message": str(exc),
        "timestamp": datetime.now().isoformat()
    }


# UI
import gradio as gr
import threading
import time

# Bihar districts list for dropdown
BIHAR_DISTRICTS = [
    "Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga", "Siwan", 
    "Begusarai", "Katihar", "Nalanda", "Rohtas", "Saran", "Samastipur",
    "Madhubani", "Purnia", "Araria", "Kishanganj", "Supaul", "Madhepura",
    "Saharsa", "Khagaria", "Munger", "Lakhisarai", "Sheikhpura", "Nawada",
    "Jamui", "Jehanabad", "Aurangabad", "Arwal", "Kaimur", "Buxar",
    "Bhojpur", "Saran", "Siwan", "Gopalganj", "East Champaran", "West Champaran",
    "Sitamarhi", "Sheohar", "Vaishali"
]

def format_workflow_output(raw_output, agent_responses=None):
    """Format workflow output for display"""
    if not raw_output:
        return "❌ No output received"
    
    if not isinstance(raw_output, str):
        raw_output = str(raw_output)
    
    lines = raw_output.split('\n')
    formatted_lines = []
    
    capture_content = False
    
    for line in lines:
        line = str(line).strip()  # Ensure line is string
        if not line:
            if capture_content:
                formatted_lines.append("")
            continue
        
        if line.startswith('✅ Workflow Summary'):
            capture_content = True
            formatted_lines.append(f"### {line}")
            continue
        
        if not capture_content:
            continue
            
        if line.startswith('=') or line.startswith('-'):
            continue
        elif line.startswith('🎯') or line.startswith('📊') or line.startswith('🤖') or line.startswith('⏰') or line.startswith('📱'):
            formatted_lines.append(f"- {line}")
        else:
            formatted_lines.append(line)

    if not capture_content or not formatted_lines:
        return "✅ Alert generated successfully - check other tabs for details"
    
    return '\n'.join(formatted_lines)


async def run_workflow_clean(request: WorkflowRequest):
    logger.info(f"Received workflow request: {request.state}, {request.district}")
    
    try:
        # Generate dynamic alert
        sample_alert = await generate_dynamic_alert(request.district, request.state)
        
        # Generate agent responses
        agents = [
            ("📱 WhatsApp Agent", whatsapp_agent.create_whatsapp_message),
            ("📱 SMS Agent", sms_agent.create_sms_message),
            ("📞 USSD Agent", ussd_agent.create_ussd_menu),
            ("🎙️ IVR Agent", ivr_agent.create_ivr_script),
            ("🤖 Telegram Agent", telegram_agent.create_telegram_message)
        ]
        
        agent_responses = {}
        successful_agents = 0
        
        for agent_name, agent_func in agents:
            try:
                response = agent_func(sample_alert)
                agent_responses[agent_name] = response
                successful_agents += 1
            except Exception as e:
                agent_responses[agent_name] = f"Error: {str(e)}"
        
        # Generate CSV
        csv_content = generate_csv_export(sample_alert, agent_responses)
        
        return {
            "status": "success",
            "csv": csv_content,
            "raw_data": {
                "state": request.state,
                "district": request.district,
                "alert_data": sample_alert,
                "agent_responses": agent_responses
            }
        }
        
    except Exception as e:
        error_msg = f"❌ Workflow failed: {str(e)}"
        logger.exception(f"Workflow error for {request.district}, {request.state}")
        
        return {
            "message": error_msg,
            "status": "error",
            "csv": "",
            "error": str(e)
        }

    
def format_agent_responses(agent_responses):
    """Create a dedicated section for agent responses"""
    if not agent_responses:
        return "No agent responses available"
    
    formatted = ["## 📱 Agent Responses", ""]
    
    for agent_name, response in agent_responses.items():
        agent_name_str = str(agent_name)
        clean_name = agent_name_str.replace("📱 ", "").replace("📞 ", "").replace("🎙️ ", "").replace("🤖 ", "")
        
        # Convert response to string for error checking
        response_str = str(response)
        if response_str.startswith('Error:'):
            formatted.append(f"### ❌ {clean_name}")
            formatted.append(f"**Error:** {response_str}")
        else:
            formatted.append(f"### ✅ {clean_name}")
            
            if isinstance(response, dict):
                for key, value in response.items():
                    key_str = str(key)
                    value_str = str(value)
                    formatted.append(f"**{key_str.title()}:**")
                    if len(value_str) > 300:
                        formatted.append(f"```\n{value_str[:300]}...\n```")
                    elif isinstance(value, list):
                        # Handle list values
                        list_str = '\n'.join([str(item) for item in value])
                        formatted.append(f"```\n{list_str}\n```")
                    else:
                        formatted.append(f"```\n{value_str}\n```")
            else:
                response_str = str(response)
                if len(response_str) > 300:
                    formatted.append(f"```\n{response_str[:300]}...\n```")
                else:
                    formatted.append(f"```\n{response_str}\n```")
        
        formatted.append("")
    
    # Ensure all items in formatted list are strings
    formatted = [str(item) for item in formatted]
    return '\n'.join(formatted)


def format_alert_summary(raw_data):
    """Format alert summary"""
    if not raw_data or 'alert_data' not in raw_data:
        return "No alert data available"
    
    alert_data = raw_data['alert_data']
    
    summary = f"""
## 🚨 Alert Summary
**📍 Location:** {alert_data['location']['village']}, {alert_data['location']['district']}, {alert_data['location']['state']}
**🌾 Crop Information:**
- **Crop:** {alert_data['crop']['name'].title()}
- **Growth Stage:** {alert_data['crop']['stage']}
- **Season:** {alert_data['crop']['season'].title()}
**🌤️ Weather Conditions:**
- **Temperature:** {alert_data['weather']['temperature']}
- **Expected Rainfall:** {alert_data['weather']['expected_rainfall']}
- **Wind Speed:** {alert_data['weather']['wind_speed']}
- **Rain Probability:** {alert_data['weather']['rain_probability']}%
**⚠️ Alert Details:**
- **Type:** {alert_data['alert']['type'].replace('_', ' ').title()}
- **Urgency:** {alert_data['alert']['urgency'].upper()}
- **AI Enhanced:** {'✅ Yes' if alert_data['alert']['ai_generated'] else '❌ No'}
**📨 Alert Message:**
{alert_data['alert']['message']}
**🎯 Action Items:**
{chr(10).join([f"- {item.replace('_', ' ').title()}" for item in alert_data['alert']['action_items']])}
"""
    return summary


def run_workflow_ui(district):
    """Run workflow directly using internal functions with clean output"""
    if not district:
        return "❌ Please select a district", "", "", gr.File(visible=False)
    
    try:
        request_obj = WorkflowRequest(state="bihar", district=district.lower())
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(run_workflow_clean(request_obj))
        
        # Extract data
        raw_data = result.get('raw_data', {})
        agent_responses = raw_data.get('agent_responses', {})
        
        alert_summary = format_alert_summary(raw_data)
        agent_details = format_agent_responses(agent_responses)
        csv_content = result.get('csv', '')
        
        # Create CSV file if content exists
        if csv_content:
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(csv_content)
                temp_filename = temp_file.name
            
            display_name = f"bihar_alert_{district.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            return alert_summary, agent_details, gr.File(value=temp_filename, visible=True, label=display_name)
        else:
            return alert_summary, agent_details, gr.File(visible=False)
            
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logger.exception(f"UI workflow error: {e}")
        return error_msg, "", "", gr.File(visible=False)



# Gradio interface
def create_gradio_interface():
    with gr.Blocks(
        title="BIHAR AgMCP - Agricultural Weather Alerts",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
        # 🌾 BIHAR AgMCP - Agricultural Weather Alert System
        
        **AI-Powered Weather Alerts for Bihar Farmers**
        
        Generate personalized weather alerts for agricultural activities in Bihar districts.
        
        ## How to Use:
        1. Select a Bihar district from the dropdown
        2. Click "Generate Weather Alert" 
        3. View the formatted results and download CSV data
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                district_input = gr.Dropdown(
                    choices=BIHAR_DISTRICTS,
                    label="📍 Select Bihar District",
                    value="Patna"
                )
                
                run_btn = gr.Button(
                    "🚀 Generate Weather Alert", 
                    variant="primary",
                    size="lg"
                )
        
        with gr.Row():
                alert_summary = gr.Markdown(
                    label="📊 Alert Summary",
                    value="Alert details will appear here..."
                )
        
        with gr.Row():
            agent_responses = gr.Markdown(
                label="📱 Agent Messages",
                value="Agent responses will appear here..."
            )
        
        csv_output = gr.File(
            label="📁 Download CSV Data",
            visible=False
        )
        
        # Connect the button 
        run_btn.click(
            run_workflow_ui,
            inputs=[district_input],
            outputs=[alert_summary, agent_responses, csv_output]
        )
    
    return demo

def run_fastapi():
    """Run FastAPI server"""
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level=LOG_LEVEL.lower()
    )

# Usage in main section:
if __name__ == "__main__":
    if os.getenv("SPACE_ID") or os.getenv("GRADIO_SERVER_NAME"):
        # HuggingFace Spaces - start FastAPI in background, Gradio in foreground
        logger.info("Starting in HuggingFace Spaces mode with UI...")
        
        # Start FastAPI in background
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        
        # Wait for FastAPI to start
        time.sleep(3)
        
        # Launch Gradio interface
        demo = create_gradio_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
    else:
        logger.info("Starting MCP Weather Server (API only)...")
        run_fastapi()
        

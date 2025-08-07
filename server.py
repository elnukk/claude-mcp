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
        # Get villages for the district
        villages_data = await geographic_tools.list_villages(state, district)
        
        # Pick a random village or use default
        village_name = f"Village in {district}"
        if "villages" in villages_data and villages_data["villages"]:
            village_name = random.choice(villages_data["villages"])
            # Avoid village name being same as district
            if village_name.lower() == district.lower() and len(villages_data["villages"]) > 1:
                other_villages = [v for v in villages_data["villages"] if v.lower() != district.lower()]
                if other_villages:
                    village_name = random.choice(other_villages)
        
        # Get coordinates for the district/village
        location_coords = [25.5941, 85.1376]  # Default to Patna
        
        # Try to get coordinates for the district first
        try:
            district_location = await geographic_tools.reverse_geocode(district)
            if "error" not in district_location and "lat" in district_location:
                location_coords = [district_location["lat"], district_location["lng"]]
        except:
            pass  # Keep default coordinates
        
        # Generate regional crop and stage
        regional_crop = get_regional_crop_for_area(district, state)
        crop_stage = get_current_crop_stage(regional_crop)
        
        # GET WEATHER DATA 
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
            
            # Higher precipitation = higher humidity estimate
            estimated_humidity = min(95, max(40, 60 + int(next_3_days_rain * 2)))
            
            real_weather = {
                "forecast_days": 3,
                "rain_probability": rain_probability,
                "expected_rainfall": f"{next_3_days_rain:.1f}mm",
                "temperature": f"{current_temp:.1f}°C",
                "humidity": f"{estimated_humidity}%",
                "wind_speed": f"{current_windspeed:.1f} km/h"
            }
            
            # Generate alert message based on weather conditions
            if next_3_days_rain > 25:
                alert_type = "heavy_rain_warning"
                urgency = "high"
                alert_message = f"Heavy rainfall ({next_3_days_rain:.1f}mm) expected in next 3 days in {district}. Delay fertilizer application. Ensure proper drainage."
                action_items = ["delay_fertilizer", "check_drainage", "monitor_crops", "prepare_harvest_protection"]
            elif next_3_days_rain > 10:
                alert_type = "moderate_rain_warning"
                urgency = "medium"
                alert_message = f"Moderate rainfall ({next_3_days_rain:.1f}mm) expected in next 3 days in {district}. Monitor soil moisture levels."
                action_items = ["monitor_soil", "check_drainage", "adjust_irrigation"]
            elif next_3_days_rain < 2 and current_temp > 35:
                alert_type = "heat_drought_warning"
                urgency = "high"
                alert_message = f"High temperature ({current_temp:.1f}°C) with minimal rainfall expected in {district}. Increase irrigation frequency."
                action_items = ["increase_irrigation", "mulch_crops", "monitor_plant_stress"]
            else:
                alert_type = "weather_update"
                urgency = "low"
                alert_message = f"Normal weather conditions expected in {district}. Temperature {current_temp:.1f}°C, rainfall {next_3_days_rain:.1f}mm."
                action_items = ["routine_monitoring", "maintain_irrigation"]
            
            logger.info(f"Real weather data retrieved for {district}: {current_temp}°C, {next_3_days_rain:.1f}mm rain")
            
        except Exception as weather_error:
            logger.error(f"Failed to get real weather data for {district}: {weather_error}")
            raise Exception(f"Unable to retrieve current weather conditions for {district}")
        
        return {
            "alert_id": f"{state.upper()[:2]}_{district.upper()[:3]}_001_{datetime.now().strftime('%Y%m%d')}",
            "timestamp": datetime.now().isoformat() + "Z",
            "location": {
                "village": village_name,
                "district": district,
                "state": state.capitalize(),
                "coordinates": location_coords
            },
            "crop": {
                "name": regional_crop,
                "stage": crop_stage,
                "planted_estimate": "2025-06-15"
            },
            "alert": {
                "type": alert_type,
                "urgency": urgency,
                "message": alert_message,
                "action_items": action_items,
                "valid_until": (datetime.now() + timedelta(days=3)).isoformat() + "Z"
            },
            "weather": real_weather,
            "data_source": "open_meteo_api"  
        }
    
    except Exception as e:
        logger.error(f"Error generating dynamic alert for {district}, {state}: {e}")
        raise Exception(f"Failed to generate weather alert for {district}: {str(e)}")


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
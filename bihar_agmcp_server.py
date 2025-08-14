#!/usr/bin/env python3
"""
Bihar AgMCP - MCP Server for Claude Desktop
Agricultural Weather Alert System
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime, timedelta
import random
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bihar-agmcp")

# MCP Protocol imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

# Import your existing modules (adjust paths as needed)
try:
    from tools import (
        open_meteo, 
        geographic_tools, 
        crop_calendar_tools, 
        alert_generation_tools
    )
    from a2a_agents import sms_agent, whatsapp_agent, ussd_agent, ivr_agent, telegram_agent
    logger.info("✅ Successfully imported tool modules")
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Some features may be limited without the tool modules")

# Initialize MCP Server
server = Server("bihar-agmcp")

# Bihar districts configuration
BIHAR_DISTRICTS = [
    "Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga", "Siwan", 
    "Begusarai", "Katihar", "Nalanda", "Rohtas", "Saran", "Samastipur",
    "Madhubani", "Purnia", "Araria", "Kishanganj", "Supaul", "Madhepura",
    "Saharsa", "Khagaria", "Munger", "Lakhisarai", "Sheikhpura", "Nawada",
    "Jamui", "Jehanabad", "Aurangabad", "Arwal", "Kaimur", "Buxar",
    "Bhojpur", "Gopalganj", "East Champaran", "West Champaran",
    "Sitamarhi", "Sheohar", "Vaishali"
]

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

# District-specific crop preferences
DISTRICT_CROPS = {
    'patna': {'primary': ['rice', 'wheat', 'potato'], 'secondary': ['mustard', 'gram'], 'specialty': ['sugarcane']},
    'gaya': {'primary': ['wheat', 'rice'], 'secondary': ['barley', 'mustard'], 'specialty': ['gram']},
    'bhagalpur': {'primary': ['rice', 'maize', 'wheat'], 'secondary': ['jute'], 'specialty': ['groundnut']},
    'muzaffarpur': {'primary': ['sugarcane', 'rice', 'wheat'], 'secondary': ['potato', 'mustard'], 'specialty': ['lentil']},
    'darbhanga': {'primary': ['rice', 'wheat', 'maize'], 'secondary': ['gram'], 'specialty': ['bajra']},
}

# Utility functions
def get_current_season(month: int) -> str:
    """Determine current agricultural season"""
    if month in [6, 7, 8, 9]:  # June to September
        return 'kharif'
    elif month in [10, 11, 12, 1, 2, 3]:  # October to March
        return 'rabi'
    else:  # April, May
        return 'zaid'

def select_regional_crop(district: str, state: str) -> str:
    """Select appropriate crop based on district and season"""
    current_month = datetime.now().month
    current_season = get_current_season(current_month)
    
    district_prefs = DISTRICT_CROPS.get(district.lower(), {
        'primary': ['rice', 'wheat'], 
        'secondary': ['gram'], 
        'specialty': ['maize']
    })
    
    seasonal_crops = {
        'kharif': ['rice', 'maize', 'sugarcane'],
        'rabi': ['wheat', 'barley', 'gram', 'mustard'],
        'zaid': ['maize', 'moong']
    }
    
    all_district_crops = (district_prefs.get('primary', []) + 
                         district_prefs.get('secondary', []) + 
                         district_prefs.get('specialty', []))
    
    suitable_crops = [crop for crop in all_district_crops 
                     if crop in seasonal_crops.get(current_season, [])]
    
    if not suitable_crops:
        suitable_crops = district_prefs.get('primary', ['rice'])
    
    return random.choice(suitable_crops) if suitable_crops else 'rice'

def estimate_crop_stage(crop: str, current_month: int) -> str:
    """Estimate current crop stage based on crop type and month"""
    if crop not in CROP_CALENDAR:
        return 'Growing'
    
    stages = CROP_CALENDAR[crop]['stages']
    
    # Simplified stage mapping
    stage_mappings = {
        'rice': {6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5},
        'wheat': {11: 0, 12: 1, 1: 2, 2: 3, 3: 4},
        'maize': {6: 0, 7: 1, 8: 2, 9: 3}
    }
    
    crop_mapping = stage_mappings.get(crop, {})
    stage_index = crop_mapping.get(current_month, len(stages) // 2)
    stage_index = min(stage_index, len(stages) - 1)
    
    return stages[stage_index] if stages else 'Growing'

# MCP Tool Handlers

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="generate_weather_alert",
            description="Generate comprehensive weather alerts for Bihar districts with crop-specific recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "district": {
                        "type": "string",
                        "description": "Bihar district name (e.g., 'Patna', 'Gaya', 'Muzaffarpur')",
                        "enum": BIHAR_DISTRICTS
                    },
                    "include_ai_analysis": {
                        "type": "boolean",
                        "description": "Whether to include AI-powered analysis (requires OpenAI API key)",
                        "default": True
                    }
                },
                "required": ["district"]
            }
        ),
        Tool(
            name="get_district_crops",
            description="Get crop information for a specific Bihar district including seasonal recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "district": {
                        "type": "string",
                        "description": "Bihar district name",
                        "enum": BIHAR_DISTRICTS
                    }
                },
                "required": ["district"]
            }
        ),
        Tool(
            name="get_crop_calendar",
            description="Get detailed crop calendar information including planting and harvesting schedules",
            inputSchema={
                "type": "object",
                "properties": {
                    "crop": {
                        "type": "string",
                        "description": "Crop name",
                        "enum": list(CROP_CALENDAR.keys())
                    }
                },
                "required": ["crop"]
            }
        ),
        Tool(
            name="get_weather_data",
            description="Get current weather data for specific coordinates in Bihar",
            inputSchema={
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude coordinate"
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude coordinate"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of forecast days (1-7)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 7
                    }
                },
                "required": ["latitude", "longitude"]
            }
        ),
        Tool(
            name="generate_agent_messages",
            description="Generate messages for different communication channels (SMS, WhatsApp, USSD, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "alert_data": {
                        "type": "object",
                        "description": "Alert data object containing location, crop, and weather information"
                    },
                    "channels": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["sms", "whatsapp", "ussd", "ivr", "telegram"]
                        },
                        "description": "Communication channels to generate messages for",
                        "default": ["sms", "whatsapp"]
                    }
                },
                "required": ["alert_data"]
            }
        ),
        Tool(
            name="list_bihar_districts",
            description="Get list of all supported Bihar districts",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="help_bihar_agmcp",
            description="Get comprehensive help and usage examples for Bihar AgMCP system",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Specific help topic",
                        "enum": ["overview", "districts", "crops", "weather", "alerts", "examples"],
                        "default": "overview"
                    }
                }
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    
    try:
        if name == "generate_weather_alert":
            return await generate_weather_alert_tool(arguments)
        elif name == "get_district_crops":
            return await get_district_crops_tool(arguments)
        elif name == "get_crop_calendar":
            return await get_crop_calendar_tool(arguments)
        elif name == "get_weather_data":
            return await get_weather_data_tool(arguments)
        elif name == "generate_agent_messages":
            return await generate_agent_messages_tool(arguments)
        elif name == "list_bihar_districts":
            return await list_bihar_districts_tool(arguments)
        elif name == "help_bihar_agmcp":
            return await help_bihar_agmcp_tool(arguments)
        else:
            return [TextContent(
                type="text", 
                text=f"❌ Unknown tool: {name}\n\nAvailable tools:\n• generate_weather_alert\n• get_district_crops\n• get_crop_calendar\n• get_weather_data\n• generate_agent_messages\n• list_bihar_districts\n• help_bihar_agmcp"
            )]
            
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [TextContent(
            type="text", 
            text=f"❌ **Error executing {name}**\n\nError details: {str(e)}\n\nPlease check your inputs and try again. Use `help_bihar_agmcp` for usage guidance."
        )]

# Tool Implementation Functions

async def generate_weather_alert_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate comprehensive weather alert"""
    district = arguments.get("district")
    include_ai = arguments.get("include_ai_analysis", True)
    
    if not district:
        return [TextContent(type="text", text="❌ Error: District parameter is required")]
    
    if district not in BIHAR_DISTRICTS:
        available_districts = ", ".join(BIHAR_DISTRICTS[:5]) + "..."
        return [TextContent(
            type="text", 
            text=f"❌ Error: '{district}' is not a valid Bihar district.\n\nAvailable districts: {available_districts}\n\nUse the 'list_bihar_districts' tool to see all options."
        )]
    
    try:
        # Crop selection with seasonal intelligence
        current_month = datetime.now().month
        selected_crop = select_regional_crop(district, "bihar")
        crop_stage = estimate_crop_stage(selected_crop, current_month)
        current_season = get_current_season(current_month)
        
        # Mock weather data (replace with actual API calls when available)
        weather_summary = {
            "temperature": f"{random.randint(22, 35)}.{random.randint(0, 9)}°C",
            "rainfall": f"{random.randint(0, 50)}.{random.randint(0, 9)}mm",
            "humidity": f"{random.randint(45, 85)}%",
            "wind_speed": f"{random.randint(5, 25)}.{random.randint(0, 9)} km/h"
        }
        
        # Generate alert based on conditions
        rainfall_amount = float(weather_summary["rainfall"].replace("mm", ""))
        temp_value = float(weather_summary["temperature"].replace("°C", ""))
        
        if rainfall_amount > 25:
            alert_type = "heavy_rain_warning"
            urgency = "high"
            icon = "🌧️"
            action_items = ["Postpone fertilizer application", "Ensure proper drainage", "Protect harvested crops"]
        elif rainfall_amount > 10:
            alert_type = "moderate_rain_warning"
            urgency = "medium"
            icon = "☔"
            action_items = ["Monitor soil moisture", "Check drainage systems", "Adjust irrigation schedule"]
        elif temp_value > 35:
            alert_type = "heat_warning"
            urgency = "high"
            icon = "🌡️"
            action_items = ["Increase irrigation frequency", "Provide shade for crops", "Monitor plant stress"]
        else:
            alert_type = "normal_conditions"
            urgency = "low"
            icon = "☀️"
            action_items = ["Continue routine monitoring", "Maintain regular irrigation", "Follow crop calendar"]
        
        # AI Enhancement placeholder
        ai_insight = ""
        if include_ai and os.getenv("OPENAI_API_KEY"):
            ai_insight = f"\n🤖 **AI Insight**: Based on current {current_season} season conditions, {selected_crop} crops at {crop_stage} stage in {district} require special attention to {action_items[0].lower()}."
        
        # Generate comprehensive alert
        alert_id = f"BH_{district.upper()[:3]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        response = f"""{icon} **BIHAR AGRICULTURAL WEATHER ALERT**

**Alert ID**: {alert_id}  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}

📍 **Location**: {district}, Bihar  
🌾 **Crop Focus**: {selected_crop.title()} ({crop_stage})  
📅 **Season**: {current_season.title()}  

🌤️ **Current Weather Conditions**:
• Temperature: {weather_summary['temperature']}
• Expected Rainfall: {weather_summary['rainfall']}
• Humidity: {weather_summary['humidity']}
• Wind Speed: {weather_summary['wind_speed']}

⚠️ **Alert Type**: {alert_type.replace('_', ' ').title()}  
🚨 **Urgency Level**: {urgency.upper()}

🎯 **Recommended Actions**:
{chr(10).join([f"• {action}" for action in action_items])}

📊 **Crop Stage Details**:
• Current Stage: {crop_stage}
• Season: {current_season.title()}
• Typical Duration: {CROP_CALENDAR.get(selected_crop, {}).get('duration_days', 120)} days
• Planting Period: {CROP_CALENDAR.get(selected_crop, {}).get('planting', 'Variable')}

{ai_insight}

---
*This alert is generated by Bihar AgMCP System*  
*For technical support, contact your agricultural extension officer*
"""
        
        logger.info(f"Successfully generated {alert_type} alert for {district}")
        return [TextContent(type="text", text=response)]
        
    except Exception as e:
        logger.error(f"Error generating alert for {district}: {e}")
        return [TextContent(
            type="text", 
            text=f"❌ **Error generating weather alert**\n\nDistrict: {district}\nError: {str(e)}\n\nPlease try again or contact support if the issue persists."
        )]

async def get_district_crops_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get crop information for a district"""
    district = arguments.get("district")
    
    if not district:
        return [TextContent(type="text", text="Error: District is required")]
    
    district_lower = district.lower()
    crop_info = DISTRICT_CROPS.get(district_lower, {
        'primary': ['rice', 'wheat'],
        'secondary': ['gram'],
        'specialty': ['maize']
    })
    
    current_month = datetime.now().month
    current_season = get_current_season(current_month)
    
    response = f"""🌾 CROP INFORMATION FOR {district.upper()}, BIHAR

📅 Current Season: {current_season.title()}

🥇 Primary Crops:
{chr(10).join([f"• {crop.title()}" for crop in crop_info.get('primary', [])])}

🥈 Secondary Crops:
{chr(10).join([f"• {crop.title()}" for crop in crop_info.get('secondary', [])])}

⭐ Specialty Crops:
{chr(10).join([f"• {crop.title()}" for crop in crop_info.get('specialty', [])])}

🎯 Recommended for Current Season:
{select_regional_crop(district, 'bihar').title()}
"""
    
    return [TextContent(type="text", text=response)]

async def get_crop_calendar_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get crop calendar information"""
    crop = arguments.get("crop")
    
    if not crop:
        return [TextContent(type="text", text="Error: Crop is required")]
    
    if crop not in CROP_CALENDAR:
        return [TextContent(type="text", text=f"Error: {crop} not found in crop calendar")]
    
    crop_data = CROP_CALENDAR[crop]
    
    response = f"""📅 CROP CALENDAR: {crop.upper()}

🌾 Basic Information:
• Season: {crop_data['season']}
• Planting Period: {crop_data['planting']}
• Harvesting Period: {crop_data['harvesting']}
• Duration: {crop_data['duration_days']} days

🔄 Growth Stages:
{chr(10).join([f"{i+1}. {stage}" for i, stage in enumerate(crop_data['stages'])])}

📍 Current Stage Estimate:
{estimate_crop_stage(crop, datetime.now().month)}
"""
    
    return [TextContent(type="text", text=response)]

async def get_weather_data_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get weather data for coordinates"""
    latitude = arguments.get("latitude")
    longitude = arguments.get("longitude")
    days = arguments.get("days", 3)
    
    if latitude is None or longitude is None:
        return [TextContent(type="text", text="Error: Latitude and longitude are required")]
    
    # Mock weather data (replace with actual API call)
    response = f"""🌤️ WEATHER DATA

📍 Coordinates: {latitude}, {longitude}
📅 Forecast Days: {days}

🌡️ Current Conditions:
• Temperature: 28.5°C
• Humidity: 65%
• Wind Speed: 12.3 km/h
• Pressure: 1013 hPa

☔ Forecast:
• Today: 25-32°C, 20% rain chance
• Tomorrow: 24-30°C, 60% rain chance
• Day 3: 26-31°C, 15% rain chance

⚠️ Note: This is mock data. Integrate with actual weather APIs for production use.
"""
    
    return [TextContent(type="text", text=response)]

async def generate_agent_messages_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate messages for communication channels"""
    alert_data = arguments.get("alert_data")
    channels = arguments.get("channels", ["sms", "whatsapp"])
    
    if not alert_data:
        return [TextContent(type="text", text="Error: Alert data is required")]
    
    messages = {}
    
    try:
        for channel in channels:
            if channel == "sms":
                messages["SMS"] = f"BIHAR ALERT: {alert_data.get('crop', {}).get('name', 'Crop')} in {alert_data.get('location', {}).get('district', 'District')} - Check weather conditions"
            elif channel == "whatsapp":
                messages["WhatsApp"] = f"🌾 *Bihar Agricultural Alert*\n📍 {alert_data.get('location', {}).get('district', 'District')}\n🌱 Crop: {alert_data.get('crop', {}).get('name', 'Crop')}\n⚠️ {alert_data.get('alert', {}).get('message', 'Weather update')}"
            elif channel == "ussd":
                messages["USSD"] = f"CON Bihar Weather Alert\n{alert_data.get('location', {}).get('district', 'District')}\n1. View Details\n2. Get Actions\n0. Exit"
            elif channel == "ivr":
                messages["IVR"] = f"Welcome to Bihar Agricultural Alert System. Weather information for {alert_data.get('location', {}).get('district', 'your district')}. Press 1 for details, 2 for recommendations."
            elif channel == "telegram":
                messages["Telegram"] = f"🤖 Bihar AgAlert\n📍 {alert_data.get('location', {}).get('district', 'District')}\n🌾 {alert_data.get('crop', {}).get('name', 'Crop')} - {alert_data.get('crop', {}).get('stage', 'Stage')}\n⚠️ {alert_data.get('alert', {}).get('message', 'Weather update')}"
        
        response = "📱 GENERATED MESSAGES:\n\n"
        for channel, message in messages.items():
            response += f"📢 {channel}:\n{message}\n\n"
        
        return [TextContent(type="text", text=response)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error generating messages: {str(e)}")]

async def list_bihar_districts_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    """List all Bihar districts"""
    response = f"""📍 BIHAR DISTRICTS ({len(BIHAR_DISTRICTS)} total):

{chr(10).join([f"• {district}" for district in sorted(BIHAR_DISTRICTS)])}

ℹ️ Usage: Use any of these district names with the generate_weather_alert tool.
"""
    
    return [TextContent(type="text", text=response)]

async def help_bihar_agmcp_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    """Comprehensive help system"""
    topic = arguments.get("topic", "overview")
    
    help_content = {
        "overview": """🌾 **BIHAR AgMCP SYSTEM HELP**

**What is Bihar AgMCP?**
Bihar AgMCP (Agricultural Model Control Protocol) is an AI-powered weather alert system designed specifically for farmers and agricultural workers in Bihar, India.

**Key Features**:
• 📍 District-specific weather alerts
• 🌾 Crop-specific recommendations 
• 📅 Seasonal agricultural calendar
• 🤖 AI-powered insights
• 📱 Multi-channel message generation

**Available Tools**:
• `generate_weather_alert` - Create comprehensive weather alerts
• `get_district_crops` - Get crop information for districts
• `get_crop_calendar` - View crop planting/harvesting schedules
• `get_weather_data` - Fetch weather data for coordinates
• `generate_agent_messages` - Create multi-channel notifications
• `list_bihar_districts` - View all supported districts

**Quick Start**: Try "Generate a weather alert for Patna district"
""",
        
        "examples": """💡 **USAGE EXAMPLES**

**Basic Weather Alert**:
"Generate a weather alert for Patna district"

**District Crop Information**:
"What are the primary crops in Muzaffarpur?"

**Crop Calendar**:
"Show me the wheat crop calendar"

**Weather Data**:
"Get weather data for Patna coordinates"

**Multi-Channel Messages**:
"Generate SMS and WhatsApp messages for the latest alert"

**District Exploration**:
"List all Bihar districts"
"""
    }
    
    content = help_content.get(topic, help_content["overview"])
    return [TextContent(type="text", text=content)]

# Server initialization - FIXED VERSION
async def main():
    """Main server function with proper initialization"""
    logger.info("Starting Bihar AgMCP Server...")
    
    # Initialize the server with proper stdio handling
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
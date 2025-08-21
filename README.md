# Bihar AgMCP - Agricultural Weather Alert System

## Table of Contents
- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Installation & Setup](#installation--setup)
- [Core Components](#core-components)
- [API Reference](#api-reference)
- [Data Models](#data-models)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## System Overview

### Purpose
The Bihar AgMCP (Agricultural Model Control Protocol) is an AI-powered weather alert system designed to provide personalized agricultural advisories to farmers in Bihar, India. The system generates location-specific weather alerts with crop-specific recommendations.

### Key Features
- **AI-Enhanced Weather Alerts**: OpenAI-powered intelligent alert generation
- **Multi-Channel Communication**: SMS, WhatsApp, USSD, IVR, and Telegram support
- **Regional Crop Intelligence**: District-specific crop recommendations based on seasons
- **Real-time Weather Data**: Integration with multiple weather APIs
- **Geographic Intelligence**: Village-level weather data and coordinates
- **Web Interface**: User-friendly Gradio interface for easy access
- **CSV Export**: Data export functionality for analysis

### Target Users
- Agricultural extension workers
- Farmers and farmer organizations
- Government agricultural departments
- Research institutions
- NGOs working in agriculture

## Architecture

### System Architecture Diagram
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Weather APIs  │    │   OpenAI API    │    │ Geographic APIs │
│                 │    │                 │    │                 │
│ • Open-Meteo    │    │ • GPT Models    │    │ • Geocoding     │
│ • Tomorrow.io   │    │ • Alert Gen     │    │ • Village Data  │
│ • OpenWeather   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   FastAPI Core  │
                    │                 │
                    │ • Alert Engine  │
                    │ • Crop Calendar │
                    │ • Workflow Mgmt │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   A2A Agents    │    │   Web Interface │    │   API Clients   │
│                 │    │                 │    │                 │
│ • SMS Agent     │    │ • Gradio UI     │    │ • MCP Protocol  │
│ • WhatsApp      │    │ • CSV Export    │    │ • REST API      │
│ • USSD/IVR      │    │ • District Sel  │    │ • JSON-RPC      │
│ • Telegram      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack
- **Backend**: FastAPI (Python 3.8+)
- **AI**: OpenAI GPT 
- **Weather Data**: Open-Meteo, Tomorrow.io, OpenWeatherMap
- **Frontend**: Gradio
- **Data Processing**: Pandas, CSV
- **Async Operations**: asyncio
- **Configuration**: dotenv
- **Logging**: Python logging

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection for API access

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bihar-agmcp
   ```

2. **Install dependencies**
   ```bash
   pip install fastapi uvicorn gradio pandas python-dotenv pydantic
   ```

3. **Create environment file**
   Create a `.env` file in the root directory:
   ```env
   # Required
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional Weather APIs
   TOMORROW_IO_API_KEY=your_tomorrow_io_key
   GOOGLE_WEATHER_API_KEY=your_google_weather_key
   OPENWEATHERMAP_API_KEY=your_openweathermap_key
   ACCUWEATHER_API_KEY=your_accuweather_key
   
   # Logging
   LOG_LEVEL=INFO
   ```

4. **Tool dependencies**
   Ensure the following modules are available:
   - `tools/` - Weather and geographic tools
   - `a2a_agents/` - Communication agents
   - `utils/` - Utility functions

### Quick Start
```bash
python main.py
```

## Core Components

### 1. Server-side

#### a. Alert Generation Engine
- **Primary Function**: `generate_dynamic_alert(district, state)`
- **Purpose**: Generates comprehensive weather alerts for specific districts.

**Process Flow:**
1. **Village Selection**: Randomly selects a village from the district
2. **Coordinate Resolution**: Gets GPS coordinates for location
3. **Crop Intelligence**: Selects appropriate crop based on:
   - Current season (Kharif/Rabi/Zaid)
   - District crop preferences
   - Regional agricultural patterns
4. **Weather Data Collection**: Fetches weather data from APIs
5. **Alert Generation**: Creates weather-based alerts
6. **AI Enhancement**: Applies OpenAI analysis (if available)

**Example Usage:**
```python
alert_data = await generate_dynamic_alert("Patna", "Bihar")
```

#### b. Crop Calendar System

**Crop Definitions**
The system includes detailed crop calendars for major Bihar crops:
- **Rice (Kharif)**: June-July planting, October-November harvest
- **Wheat (Rabi)**: November-December planting, March-April harvest
- **Maize (Kharif/Zaid)**: Dual season crop
- **Sugarcane (Annual)**: February-March planting
- **Mustard (Rabi)**: October-November planting

**District Crop Mapping**
Each district has specific crop preferences:
```python
DISTRICT_CROPS = {
    'patna': {
        'primary': ['rice', 'wheat', 'potato'], 
        'secondary': ['mustard', 'gram'], 
        'specialty': ['sugarcane']
    },
    # ... other districts
}
```

#### c. Weather Integration

**Supported Weather APIs:**
- **Open-Meteo** (Primary): Free, reliable weather data
- **Tomorrow.io**: Professional weather services
- **OpenWeatherMap**: Popular weather API
- **AccuWeather**: Commercial weather data
- **Google Weather**: Google's weather services

**Weather Alert Types:**
- **Heavy Rain Warning**: >25mm rainfall expected
- **Moderate Rain Warning**: 10-25mm rainfall expected
- **Heat/Drought Warning**: High temperature + low rainfall
- **Cold Warning**: Temperature <10°C
- **High Wind Warning**: Wind speed >30 km/h
- **Weather Update**: Normal conditions

### 2. Client-side

#### A2A Communication Agents

**SMS Agent**
Creates concise SMS messages (160 characters max):
```python
def create_sms_message(alert_data):
    return f"BIHAR ALERT: {crop} in {location} - {weather_summary}"
```

**WhatsApp Agent**
Generates rich WhatsApp messages with emojis and formatting:
```python
def create_whatsapp_message(alert_data):
    return {
        "text": formatted_message,
        "buttons": action_buttons
    }
```

**USSD Agent**
Creates interactive USSD menu structures:
```python
def create_ussd_menu(alert_data):
    return {
        "menu": "CON Weather Alert...\n1. Details\n2. Actions"
    }
```

**IVR Agent**
Generates voice script for phone systems:
```python
def create_ivr_script(alert_data):
    return {
        "script": "Welcome to Bihar weather alert...",
        "menu_options": [...]
    }
```

**Telegram Agent**
Creates Telegram messages with inline keyboards:
```python
def create_telegram_message(alert_data):
    return {
        "text": message,
        "reply_markup": inline_keyboard
    }
```

## API Reference

### Core Endpoints

#### 1. Workflow Execution
```http
POST /api/run-workflow
```

**Request Body:**
```json
{
    "state": "bihar",
    "district": "patna"
}
```

**Response:**
```json
{
    "message": "workflow_results",
    "status": "success",
    "csv": "csv_data",
    "raw_data": {
        "alert_data": {...},
        "agent_responses": {...}
    }
}
```

#### 2. Health Check
```http
GET /api/health
```

**Response:**
```json
{
    "status": "healthy",
    "openai_available": true,
    "timestamp": "2025-08-13T10:30:00"
}
```

#### 3. MCP Tool Execution
```http
POST /mcp
```

**Request Body:**
```json
{
    "tool": "get_current_weather",
    "parameters": {
        "latitude": 25.5941,
        "longitude": 85.1376
    }
}
```

#### 4. Geographic Data
```http
GET /api/districts/{state}
GET /api/villages/{state}/{district}
```

#### 5. Weather Data
```http
GET /api/weather/{latitude}/{longitude}
```

### A2A Agent Endpoints
- **SMS**: `POST /a2a/sms`
- **WhatsApp**: `POST /a2a/whatsapp`
- **USSD**: `POST /a2a/ussd`
- **IVR**: `POST /a2a/ivr`
- **Telegram**: `POST /a2a/telegram`

## Data Models

### AlertRequest
```python
class AlertRequest(BaseModel):
    alert_json: dict
```

### WorkflowRequest
```python
class WorkflowRequest(BaseModel):
    state: str
    district: str
```

### MCPRequest
```python
class MCPRequest(BaseModel):
    tool: str
    parameters: dict
```

### Alert Data Structure
```json
{
    "alert_id": "BH_PAT_VIL_20250813_103000",
    "timestamp": "2025-08-13T10:30:00Z",
    "location": {
        "village": "Danapur",
        "district": "Patna",
        "state": "Bihar",
        "coordinates": [25.5941, 85.1376],
        "coordinates_source": "village_danapur"
    },
    "crop": {
        "name": "rice",
        "stage": "Tillering",
        "season": "kharif",
        "planted_estimate": "2025-06-15"
    },
    "alert": {
        "type": "heavy_rain_warning",
        "urgency": "high",
        "message": "Heavy rainfall expected...",
        "action_items": ["delay_fertilizer", "check_drainage"],
        "ai_generated": true
    },
    "weather": {
        "temperature": "28.5°C",
        "expected_rainfall": "35.2mm",
        "wind_speed": "15.3 km/h",
        "rain_probability": "85%"
    }
}
```

## Configuration

### Environment Variables

#### Required
- `OPENAI_API_KEY`: OpenAI API key for AI features

#### Optional
- `TOMORROW_IO_API_KEY`: Tomorrow.io weather API
- `GOOGLE_WEATHER_API_KEY`: Google Weather API
- `OPENWEATHERMAP_API_KEY`: OpenWeatherMap API
- `ACCUWEATHER_API_KEY`: AccuWeather API
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### CORS Configuration
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mcp-ui.vercel.app", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)
```

### Bihar Districts Configuration
The system supports 39 Bihar districts:
- Patna, Gaya, Bhagalpur, Muzaffarpur, Darbhanga
- Siwan, Begusarai, Katihar, Nalanda, Rohtas
- And 29 others...

## Deployment

### Local Development
```bash
python main.py
```
- FastAPI server: http://localhost:8000
- Gradio interface: http://localhost:7860

### Production Deployment

#### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000 7860

CMD ["python", "main.py"]
```

#### HuggingFace Spaces
The application automatically detects HuggingFace Spaces environment and configures accordingly:
```python
if os.getenv("SPACE_ID") or os.getenv("GRADIO_SERVER_NAME"):
    # HuggingFace Spaces mode
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

## Troubleshooting

### Common Issues

#### 1. OpenAI API Key Missing
- **Error**: `OpenAI API key not found - AI features will be limited`
- **Solution**: Add `OPENAI_API_KEY` to `.env` file

#### 2. Weather API Failures
- **Error**: `Failed to get weather data`
- **Solutions**:
  - Check internet connectivity
  - Verify API keys in `.env`
  - Check API quotas and limits

#### 3. Geographic Data Issues
- **Error**: `District 'xyz' not found in Bihar`
- **Solutions**:
  - Use correct district spelling
  - Check district name in `BIHAR_DISTRICTS` list
  - Verify state parameter is "bihar"

#### 4. Import Errors
- **Error**: `ModuleNotFoundError: No module named 'tools'`
- **Solution**: Ensure all required modules are in the project directory:
  - `tools/`
  - `a2a_agents/`
  - `utils/`

### Logging and Debugging

#### Enable Debug Logging
```bash
export LOG_LEVEL=DEBUG
python main.py
```

#### Check Log Output
```python
logger.info(f"Selected village: {village}")
logger.error(f"Weather API error: {error}")
```

### Performance Optimization

#### 1. API Response Caching
Consider implementing Redis caching for weather data:
```python
# Pseudo-code
@cache.cached(timeout=300)  # 5 minute cache
async def get_weather_data(lat, lon):
    return await weather_api.fetch(lat, lon)
```

#### 2. Database Integration
For production, consider using PostgreSQL/MongoDB:
```python
# Store alert history
await db.alerts.insert_one(alert_data)
```

## Contributing

### Development Guidelines

#### 1. Code Style
- Follow PEP 8 Python style guide
- Use type hints for function parameters
- Add docstrings for all functions
- Maximum line length: 100 characters

#### 2. Testing
```python
import pytest

def test_crop_selection():
    crop = select_regional_crop("patna", "bihar")
    assert crop in ["rice", "wheat", "potato"]

async def test_weather_api():
    result = await open_meteo.get_current_weather(25.5941, 85.1376)
    assert "temperature" in result
```

#### 3. Documentation
- Update this documentation for new features
- Add inline comments for complex logic
- Create examples for new API endpoints

#### 4. Version Control
```bash
git checkout -b feature/new-weather-provider
git commit -m "Add new weather API integration"
git push origin feature/new-weather-provider
```

### Adding New Features

#### 1. New Weather Provider
```python
# tools/new_weather_api.py
async def get_current_weather(latitude, longitude, api_key):
    # Implementation
    return weather_data
```

#### 2. New Communication Channel
```python
# a2a_agents/slack_agent.py
def create_slack_message(alert_data):
    return {
        "text": message,
        "attachments": []
    }
```

#### 3. New Crop Types
```python
CROP_CALENDAR["new_crop"] = {
    "season": "Rabi",
    "planting": "December",
    "harvesting": "May",
    "duration_days": 150,
    "stages": ["Sowing", "Growth", "Harvest"]
}
```

---

**Last Updated**: August 13, 2025

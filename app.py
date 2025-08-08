import gradio as gr
import subprocess
import threading
import time
import requests
import json
from datetime import datetime
import os

# Configuration
MCP_SERVER_PORT = 8001
MCP_SERVER_URL = "https://elanuk-mcp-hf.hf.space/"

# Bihar districts list
BIHAR_DISTRICTS = [
    "Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga", "Siwan", 
    "Begusarai", "Katihar", "Nalanda", "Rohtas", "Saran", "Samastipur",
    "Madhubani", "Purnia", "Araria", "Kishanganj", "Supaul", "Madhepura",
    "Saharsa", "Khagaria", "Munger", "Lakhisarai", "Sheikhpura", "Nawada",
    "Jamui", "Jehanabad", "Aurangabad", "Arwal", "Kaimur", "Buxar",
    "Bhojpur", "Saran", "Siwan", "Gopalganj", "East Champaran", "West Champaran",
    "Sitamarhi", "Sheohar", "Vaishali"
]

def start_mcp_server():
    """Start the MCP server in background"""
    try:
        print("🚀 Starting MCP Server...")
        # Set environment variable for the server port
        env = os.environ.copy()
        env["PORT"] = str(MCP_SERVER_PORT)
        
        process = subprocess.Popen(
            ["python", "mcp_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(15)
        
        # Check if server is running
        try:
            response = requests.get(f"{MCP_SERVER_URL}/api/health", timeout=5)
            if response.status_code == 200:
                print("✅ MCP Server started successfully!")
                return process
        except:
            pass
            
        print("⚠️ Server may still be starting...")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start MCP server: {e}")
        return None

def format_workflow_output(raw_output):
    """Format the workflow output for better display"""
    if not raw_output:
        return "❌ No output received"
    
    lines = raw_output.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append("")
            continue
            
        if line.startswith('🌾') and 'Workflow' in line:
            formatted_lines.append(f"## {line}")
        elif line.startswith('=') or line.startswith('-'):
            continue
        elif line.startswith('🌤️') or line.startswith('✅ Workflow'):
            formatted_lines.append(f"### {line}")
        elif line.startswith('📱') or line.startswith('📞') or line.startswith('🎙️') or line.startswith('🤖'):
            formatted_lines.append(f"#### {line}")
        elif line.startswith('✅') or line.startswith('❌'):
            formatted_lines.append(f"- {line}")
        elif line.startswith('   '):
            formatted_lines.append(f"  {line.strip()}")
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def format_alert_summary(raw_data):
    """Create a formatted summary of the alert data"""
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

def test_mcp_workflow(district):
    """Test the MCP workflow for a given district"""
    if not district:
        return "❌ Please select a district", "", ""
    
    try:
        payload = {
            "state": "bihar",
            "district": district.lower()
        }
        
        response = requests.post(
            f"{MCP_SERVER_URL}/api/run-workflow",
            json=payload,
            timeout=60  # Increased timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            
            workflow_output = format_workflow_output(result.get('message', ''))
            alert_summary = format_alert_summary(result.get('raw_data', {}))
            csv_content = result.get('csv', '')
            
            return workflow_output, alert_summary, csv_content
            
        else:
            error_msg = f"❌ Server Error ({response.status_code}): {response.text}"
            return error_msg, "", ""
            
    except requests.exceptions.Timeout:
        return "⏰ Request timed out. The server might be processing...", "", ""
    except requests.exceptions.ConnectionError:
        return f"🔌 Connection Error: Cannot reach MCP server. Is it running?", "", ""
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""

def check_server_health():
    """Check if the MCP server is running"""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/api/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return f"✅ Server Online | OpenAI: {'✅' if data.get('openai_available') else '❌'} | Time: {data.get('timestamp', 'N/A')}"
        else:
            return f"⚠️ Server responded with status {response.status_code}"
    except Exception as e:
        return f"❌ Server Offline: {str(e)}"

# Start server in background thread
print("🔧 Initializing BIHAR AgMCP...")
server_process = None

def start_server_thread():
    global server_process
    server_process = start_mcp_server()

server_thread = threading.Thread(target=start_server_thread, daemon=True)
server_thread.start()

# Create Gradio interface
with gr.Blocks(
    title="BIHAR AgMCP - Agricultural Weather Alerts",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px;
        margin: auto;
    }
    """
) as demo:
    
    gr.Markdown("""
    # 🌾 BIHAR AgMCP - Agricultural Weather Alert System
    
    **AI-Powered Weather Alerts for Bihar Farmers**
    
    This system generates personalized weather alerts for agricultural activities in Bihar districts.
    
    ## 📋 How to Use:
    1. **Wait for Server**: Ensure server status shows "Online" below
    2. **Select District**: Choose a Bihar district from the dropdown  
    3. **Run Workflow**: Click the button to generate weather alerts
    4. **View Results**: See formatted workflow output and alert summary
    5. **Download Data**: Get CSV export of the alert data
    
    The system will automatically:
    - Select a random village in the district
    - Choose appropriate crops based on season and region  
    - Generate weather-based agricultural alerts
    - Create messages for multiple communication channels
    """)
    
    # Server status
    with gr.Row():
        server_status = gr.Textbox(
            label="🔧 Server Status", 
            value="🔄 Starting server...",
            interactive=False,
            container=True
        )
        
        refresh_btn = gr.Button("🔄 Check Status", size="sm")
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=1):
            district_input = gr.Dropdown(
                choices=BIHAR_DISTRICTS,
                label="📍 Select Bihar District",
                placeholder="Choose a district...",
                value="Patna"
            )
            
            run_btn = gr.Button(
                "🚀 Generate Weather Alert", 
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("""
            ### 💡 What happens next?
            - Weather data collection from multiple sources
            - Intelligent crop stage estimation  
            - AI-powered alert generation
            - Multi-channel message creation (SMS, WhatsApp, etc.)
            - Comprehensive CSV data export
            """)
    
    # Results section
    with gr.Row():
        with gr.Column(scale=2):
            workflow_output = gr.Markdown(
                label="📋 Workflow Output",
                value="Server is starting... Please wait and check server status above."
            )
        
        with gr.Column(scale=1):
            alert_summary = gr.Markdown(
                label="📊 Alert Summary", 
                value="Alert details will appear here after running workflow..."
            )
    
    # CSV export
    with gr.Row():
        csv_output = gr.File(
            label="📁 Download CSV Data",
            visible=False
        )
    
    # Event handling
    def run_workflow_with_csv(district):
        workflow, summary, csv_content = test_mcp_workflow(district)
        
        if csv_content and not csv_content.startswith("Error"):
            filename = f"bihar_alert_{district.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(csv_content)
            return workflow, summary, gr.File(value=filename, visible=True)
        else:
            return workflow, summary, gr.File(visible=False)
    
    # Connect events
    refresh_btn.click(check_server_health, outputs=server_status)
    run_btn.click(
        run_workflow_with_csv,
        inputs=[district_input], 
        outputs=[workflow_output, alert_summary, csv_output]
    )
    
    # Auto-refresh server status after a delay
    def auto_check_status():
        time.sleep(20)  # Wait 20 seconds
        return check_server_health()
    
    demo.load(auto_check_status, outputs=server_status)
    
    # Footer
    gr.Markdown("""
    ---
    
    ### 🔗 System Information:
    - **State Coverage**: Bihar (38+ districts)
    - **Crops Supported**: Rice, Wheat, Maize, Sugarcane, Mustard, and more
    - **Weather Sources**: Open-Meteo API with AI enhancement
    - **Communication Channels**: SMS, WhatsApp, USSD, IVR, Telegram
    
    *Built with MCP (Model Context Protocol) for agricultural intelligence*
    """)

# Launch the interface
if __name__ == "__main__":
    print("🌾 Launching BIHAR AgMCP Interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
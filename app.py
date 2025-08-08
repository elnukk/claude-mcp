import gradio as gr
import requests
from datetime import datetime

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

def format_workflow_output(raw_output):
    """Format the workflow output for display"""
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

def run_workflow(district):
    """Run the workflow and return results"""
    if not district:
        return "❌ Please select a district", "", ""
    
    try:
        payload = {
            "state": "bihar",
            "district": district.lower()
        }
        
        # Call your existing endpoint directly
        response = requests.post(
            "http://localhost:8000/api/run-workflow",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            workflow_output = format_workflow_output(result.get('message', ''))
            alert_summary = format_alert_summary(result.get('raw_data', {}))
            csv_content = result.get('csv', '')
            
            # Create CSV file if content exists
            if csv_content:
                filename = f"bihar_alert_{district.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(csv_content)
                return workflow_output, alert_summary, gr.File(value=filename, visible=True)
            else:
                return workflow_output, alert_summary, gr.File(visible=False)
                
        else:
            error_msg = f"❌ Server Error ({response.status_code})"
            return error_msg, "", gr.File(visible=False)
            
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, "", gr.File(visible=False)

# Create Gradio interface
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
        with gr.Column(scale=2):
            workflow_output = gr.Markdown(
                label="📋 Workflow Output",
                value="Select a district and click the button to generate alerts..."
            )
        
        with gr.Column(scale=1):
            alert_summary = gr.Markdown(
                label="📊 Alert Summary",
                value="Alert details will appear here..."
            )
    
    csv_output = gr.File(
        label="📁 Download CSV Data",
        visible=False
    )
    
    # Connect the button
    run_btn.click(
        run_workflow,
        inputs=[district_input],
        outputs=[workflow_output, alert_summary, csv_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
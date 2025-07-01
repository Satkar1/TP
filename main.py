import os
import re
import yaml
import json
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu
import folium
from streamlit_folium import folium_static
import plotly.express as px
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Load configuration
CONFIG_PATH = Path("config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Set page config
st.set_page_config(
    page_title=config["app"]["name"],
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

class TravelPlanner:
    """Main application class for the AI Travel Planner"""
    
    def __init__(self):
        self._load_environment()
        self._setup_llm()
        self._setup_ui()
        self._setup_session_state()
        
    def _load_environment(self):
        """Load required environment variables"""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.weather_api_key = os.getenv("OPENWEATHER_API_KEY")
        
        if not self.gemini_api_key:
            st.error("GEMINI_API_KEY not found in .env file.")
            st.stop()
    
    def _setup_llm(self):
        """Initialize the Gemini model"""
        try:
            genai.configure(api_key=self.gemini_api_key)
            available_models = [m.name for m in genai.list_models() 
                              if 'generateContent' in m.supported_generation_methods]
            
            # Use gemini-2.0-flash-exp as specified
            self.model_name = "gemini-2.0-flash-exp"
                
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.gemini_api_key,
                temperature=0.7,
                max_output_tokens=2048
            )
        except Exception as e:
            st.error(f"Error initializing Gemini model: {e}")
            st.stop()
    
    def _setup_ui(self):
        """Setup the UI theme and styles"""
        primary_color = config["ui"]["theme"]["primary_color"]
        secondary_color = config["ui"]["theme"]["secondary_color"]
        
        st.markdown(f"""
        <style>
            .stApp {{
                background-color: {config["ui"]["theme"]["background_color"]};
            }}
            .css-1d391kg {{
                padding-top: 3.5rem;
            }}
            h1 {{
                color: {primary_color};
            }}
            h2 {{
                color: {secondary_color};
            }}
            .stButton>button {{
                background-color: {primary_color};
                color: white;
                border-radius: 8px;
                padding: 0.5rem 1rem;
            }}
            .stButton>button:hover {{
                background-color: {secondary_color};
                color: white;
            }}
            .stTextInput>div>div>input {{
                border-radius: 8px;
                padding: 0.5rem;
            }}
            .stSelectbox>div>div>div {{
                border-radius: 8px;
                padding: 0.5rem;
            }}
            .tab-content {{
                padding: 1rem;
                border-radius: 8px;
                background-color: white;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            }}
            .recommendation-card {{
                padding: 1rem;
                border-radius: 8px;
                background-color: white;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            }}
        </style>
        """, unsafe_allow_html=True)
    
    def _setup_session_state(self):
        """Initialize session state variables"""
        if 'travel_data' not in st.session_state:
            st.session_state.travel_data = None
        if 'destination_info' not in st.session_state:
            st.session_state.destination_info = None
        if 'weather_data' not in st.session_state:
            st.session_state.weather_data = None
        if 'saved_trips' not in st.session_state:
            st.session_state.saved_trips = []
    
    def _get_weather(self, city: str) -> Optional[Dict]:
        """Get weather data for a city"""
        if not self.weather_api_key or not config["features"]["enable_weather"]:
            return None
            
        try:
            url = f"{config['api']['openweather']['base_url']}/weather?q={city}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.warning(f"Could not fetch weather data: {e}")
            return None
    
    def _generate_travel_prompt(self, source: str, destination: str) -> str:
        """Generate the prompt for travel recommendations"""
        return f"""
        You are an expert travel planner assistant. Provide comprehensive travel options from {source} to {destination}. 
        Include all transportation modes that are realistically available between these locations.
        
        Present the information in this exact structured markdown table format:
        
        | Mode | Price (USD) | Duration | Comfort (1-5) | Directness | Notes |
        |------|------------|----------|--------------|------------|-------|
        | Flight | [price] | [duration] | [comfort] | [direct/indirect] | [notes] |
        | Train | [price] | [duration] | [comfort] | [direct/indirect] | [notes] |
        | Bus | [price] | [duration] | [comfort] | [direct/indirect] | [notes] |
        | Rental Car | [price] | [duration] | [comfort] | [direct] | [notes] |
        | Taxi/Uber | [price] | [duration] | [comfort] | [direct] | [notes] |
        
        Rules:
        1. Only include modes that are realistically available between these cities
        2. Provide realistic estimates for price and duration
        3. If a mode is not available, write "Not available" in the Notes column
        4. Keep notes brief (1-2 sentences max)
        
        After the table, provide a brief packing list (3-5 items per category) based on:
        - The destination's typical weather
        - Common activities there
        - Cultural considerations
        
        Categories:
        - Clothing
        - Essentials
        - Documents
        """
    
    def _parse_travel_response(self, response: str) -> Dict:
        """Parse the LLM response into structured data"""
        result = {
            "travel_options": [],
            "packing": ""
        }
        
        # Parse travel options table
        table_match = re.search(r"\|.*Mode.*\|.*Price.*\|.*Duration.*\|.*Comfort.*\|.*Directness.*\|.*Notes.*\|([\s\S]+?)\n\n", response)
        if table_match:
            table_text = table_match.group(1)
            rows = [row.strip() for row in table_text.split('\n') if row.strip()]
            
            for row in rows:
                if not row.startswith('|'):
                    continue
                    
                cells = [cell.strip() for cell in row.split('|')[1:-1]]
                if len(cells) >= 6:
                    option = {
                        "Mode": cells[0],
                        "Price (USD)": cells[1],
                        "Duration": cells[2],
                        "Comfort (1-5)": cells[3],
                        "Directness": cells[4],
                        "Notes": cells[5]
                    }
                    result["travel_options"].append(option)
        
        # Parse packing list
        packing_match = re.search(r"Categories:\s*([\s\S]+)$", response)
        if packing_match:
            result["packing"] = packing_match.group(1).strip()
        
        return result
    
    def _display_travel_options(self, options: List[Dict]):
        """Display travel options in an interactive way"""
        st.subheader("Travel Options")
        
        if not options:
            st.warning("No travel options available.")
            return
        
        # Filter out unavailable options
        available_options = [opt for opt in options if "not available" not in opt.get("Notes", "").lower()]
        
        if not available_options:
            st.warning("No available travel options found between these locations.")
            return
        
        df = pd.DataFrame(available_options)
        
        # Convert price to numeric for sorting
        df['Numeric Price'] = df['Price (USD)'].str.extract(r'(\d+)').astype(float)
        df = df.sort_values('Numeric Price')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(df.drop(columns=['Numeric Price']), hide_index=True, use_container_width=True)
        
        with col2:
            # Price comparison chart
            fig_price = px.bar(
                df, 
                x='Mode', 
                y='Numeric Price', 
                title='Price Comparison',
                color='Mode',
                labels={'Numeric Price': 'Price (USD)'}
            )
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Duration comparison chart
            fig_duration = px.bar(
                df, 
                x='Mode', 
                y='Duration', 
                title='Duration Comparison',
                color='Mode'
            )
            st.plotly_chart(fig_duration, use_container_width=True)
    
    def _display_packing_list(self, packing: str):
        """Display concise packing list"""
        if not packing or not config["features"]["enable_packing"]:
            return
            
        st.subheader("Packing Suggestions")
        
        # Split into categories
        categories = re.split(r"\n\s*-\s*", packing)
        for category in categories:
            if not category.strip():
                continue
                
            category_name, *items = category.split('\n')
            with st.expander(f"📦 {category_name.strip()}"):
                for item in items:
                    if item.strip():
                        st.write(f"- {item.strip()}")
    
    def run(self):
        """Run the main application"""
        # Sidebar
        with st.sidebar:
            st.title(config["app"]["name"])
            st.write(config["app"]["description"])
            
            # Navigation
            selected = option_menu(
                menu_title=None,
                options=["Plan Trip", "Saved Trips", "About"],
                icons=["compass", "bookmark", "info-circle"],
                default_index=0
            )
            
            if selected == "About":
                self._display_about()
                return
        
        # Main content
        st.title("✈️ AI Travel Planner Pro")
        
        # Tab layout
        tab1, tab2 = st.tabs(["Plan Your Trip", "Trip Tools"])
        
        with tab1:
            with st.form("travel_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    source = st.text_input("Departure City", placeholder="New York")
                
                with col2:
                    destination = st.text_input("Destination City", placeholder="Paris")
                
                submitted = st.form_submit_button("Generate Travel Plan")
                
                if submitted:
                    if not source or not destination:
                        st.error("Please enter both departure and destination cities.")
                    else:
                        with st.spinner("Generating travel recommendations..."):
                            try:
                                # Generate travel recommendations
                                prompt = self._generate_travel_prompt(source, destination)
                                chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(prompt))
                                response = chain.run({})
                                
                                # Parse response
                                parsed_data = self._parse_travel_response(response)
                                st.session_state.travel_data = parsed_data
                                
                                st.success("Travel plan generated successfully!")
                            except Exception as e:
                                st.error(f"Error generating travel plan: {e}")
            
            # Display results if available
            if st.session_state.travel_data:
                self._display_travel_options(st.session_state.travel_data['travel_options'])
                self._display_packing_list(st.session_state.travel_data['packing'])
        
        with tab2:
            st.header("Trip Tools")
            
            # Travel tips
            with st.expander("💡 Travel Tips"):
                st.write("""
                - Always check visa requirements before booking
                - Make copies of important documents
                - Notify your bank of travel plans
                - Check local customs and etiquette
                - Purchase travel insurance
                """)
    
    def _display_about(self):
        """Display about information in sidebar"""
        st.sidebar.header("About")
        st.sidebar.write(f"**Version:** {config['app']['version']}")
        st.sidebar.write("""
        This application uses Google's Gemini AI to provide comprehensive travel planning assistance.
        """)
        
        st.sidebar.subheader("Tech Stack")
        st.sidebar.write("""
        - Python
        - Streamlit
        - LangChain
        - Google Gemini
        - Plotly
        """)

# Run the application
if __name__ == "__main__":
    app = TravelPlanner()
    app.run()

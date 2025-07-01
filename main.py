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
            
            if 'models/gemini-1.5-pro' not in available_models:
                st.warning("Gemini 1.5 Pro not available, falling back to Gemini Flash")
                self.model_name = "gemini-1.5-flash"
            else:
                self.model_name = "gemini-1.5-pro"
                
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.gemini_api_key,
                temperature=config["api"]["gemini"]["temperature"],
                max_output_tokens=config["api"]["gemini"]["max_tokens"]
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
        Include all relevant details and present the information in structured markdown format.
        
        ## Travel Options
        
        | Mode | Price (USD) | Duration | Comfort (1-5) | Directness | Notes |
        |------|------------|----------|--------------|------------|-------|
        | Flight | | | | | |
        | Train | | | | | |
        | Bus | | | | | |
        | Rental Car | | | | | |
        | Taxi/Uber | | | | | |
        
        Fill in the table with accurate estimates. Include:
        - Price range in USD
        - Estimated duration in hours/minutes
        - Comfort level (1-5, 5 being most comfortable)
        - Directness (Direct/Indirect)
        - Any important notes
        
        ## Destination Information
        
        Provide detailed information about {destination} including:
        
        ### Top Attractions (5)
        Name | Description | Entry Fee | Best Time to Visit
        ---- | ----------- | --------- | ------------------
        
        ### Accommodation Options (5)
        Name | Type | Price Range | Rating | Distance from Center
        ---- | ---- | ----------- | ------ | --------------------
        
        ### Dining Recommendations (5)
        Name | Cuisine | Price Range | Rating | Specialty
        ---- | ------- | ----------- | ------ | ---------
        
        ### Local Tips
        - Best time to visit
        - Cultural norms to be aware of
        - Must-try local dishes
        - Transportation tips
        - Safety advice
        
        ## Packing Suggestions
        Based on:
        - Current season
        - Local weather
        - Cultural norms
        - Planned activities
        
        Provide a categorized packing list with essentials.
        """
    
    def _generate_packing_list(self, destination: str, weather_data: Optional[Dict] = None) -> str:
        """Generate packing list based on destination and weather"""
        prompt = f"""
        Create a detailed packing list for a trip to {destination}. Consider:
        - Current weather: {weather_data['weather'][0]['description'] if weather_data else 'unknown'}
        - Temperature: {weather_data['main']['temp'] if weather_data else 'unknown'}°C
        - Typical activities (sightseeing, hiking, etc.)
        - Cultural norms (modest clothing, etc.)
        
        Organize the list into categories:
        
        ### Clothing
        - 
        
        ### Toiletries
        - 
        
        ### Electronics
        - 
        
        ### Documents
        - 
        
        ### Miscellaneous
        - 
        
        Provide specific recommendations based on the destination's characteristics.
        """
        
        try:
            chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(prompt))
            return chain.run({})
        except Exception as e:
            return f"Could not generate packing list: {e}"
    
    def _parse_travel_response(self, response: str) -> Dict:
        """Parse the LLM response into structured data"""
        result = {
            "travel_options": [],
            "attractions": [],
            "accommodations": [],
            "dining": [],
            "tips": "",
            "packing": ""
        }
        
        # Parse travel options
        travel_match = re.search(r"## Travel Options\n([\s\S]+?)##", response)
        if travel_match:
            travel_table = travel_match.group(1)
            result["travel_options"] = self._parse_markdown_table(travel_table)
        
        # Parse attractions
        attractions_match = re.search(r"### Top Attractions \(5\)\n([\s\S]+?)\n\n###", response)
        if attractions_match:
            attractions_table = attractions_match.group(1)
            result["attractions"] = self._parse_markdown_table(attractions_table)
        
        # Parse accommodations
        accommodations_match = re.search(r"### Accommodation Options \(5\)\n([\s\S]+?)\n\n###", response)
        if accommodations_match:
            accommodations_table = accommodations_match.group(1)
            result["accommodations"] = self._parse_markdown_table(accommodations_table)
        
        # Parse dining
        dining_match = re.search(r"### Dining Recommendations \(5\)\n([\s\S]+?)\n\n###", response)
        if dining_match:
            dining_table = dining_match.group(1)
            result["dining"] = self._parse_markdown_table(dining_table)
        
        # Parse tips
        tips_match = re.search(r"### Local Tips\n([\s\S]+?)\n\n##", response)
        if tips_match:
            result["tips"] = tips_match.group(1).strip()
        
        # Parse packing
        packing_match = re.search(r"## Packing Suggestions\n([\s\S]+)$", response)
        if packing_match:
            result["packing"] = packing_match.group(1).strip()
        
        return result
    
    def _parse_markdown_table(self, table_text: str) -> List[Dict]:
        """Parse markdown table into list of dictionaries"""
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        if len(lines) < 2:
            return []
        
        headers = [h.strip() for h in lines[0].split('|')[1:-1]]
        data = []
        
        for line in lines[2:]:
            values = [v.strip() for v in line.split('|')[1:-1]]
            if len(values) == len(headers):
                data.append(dict(zip(headers, values)))
        
        return data
    
    def _display_travel_options(self, options: List[Dict]):
        """Display travel options in an interactive way"""
        st.subheader("Travel Options")
        
        if not options:
            st.warning("No travel options available.")
            return
        
        df = pd.DataFrame(options)
        
        # Convert price to numeric for sorting
        df['Price (USD)'] = df['Price (USD)'].str.extract(r'(\d+)').astype(float)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(df, hide_index=True, use_container_width=True)
        
        with col2:
            # Price comparison chart
            fig_price = px.bar(
                df, 
                x='Mode', 
                y='Price (USD)', 
                title='Price Comparison',
                color='Mode'
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
    
    def _display_destination_map(self, destination: str):
        """Display a map centered on the destination"""
        if not config["features"]["enable_maps"]:
            return
            
        try:
            # Get coordinates (simplified - in production would use geocoding API)
            m = folium.Map(location=[20, 0], zoom_start=2)
            folium.Marker(
                location=[20, 0],  # Placeholder - would use real coordinates
                popup=destination,
                tooltip=destination
            ).add_to(m)
            
            folium_static(m, width=700, height=400)
        except Exception as e:
            st.warning(f"Could not display map: {e}")
    
    def _display_weather(self, weather_data: Optional[Dict]):
        """Display weather information"""
        if not weather_data or not config["features"]["enable_weather"]:
            return
            
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Temperature", f"{weather_data['main']['temp']}°C")
        
        with col2:
            st.metric("Conditions", weather_data['weather'][0]['description'].title())
        
        with col3:
            icon_url = config["api"]["openweather"]["icon_url"].format(
                icon=weather_data['weather'][0]['icon']
            )
            st.image(icon_url, width=50)
    
    def _display_attractions(self, attractions: List[Dict]):
        """Display top attractions"""
        st.subheader("Top Attractions")
        
        if not attractions:
            st.warning("No attraction information available.")
            return
        
        for i, attraction in enumerate(attractions, 1):
            with st.expander(f"{i}. {attraction.get('Name', '')}"):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.write(f"**Entry Fee:** {attraction.get('Entry Fee', 'Unknown')}")
                    st.write(f"**Best Time:** {attraction.get('Best Time to Visit', 'Unknown')}")
                
                with col2:
                    st.write(attraction.get('Description', 'No description available.'))
    
    def _display_accommodations(self, accommodations: List[Dict]):
        """Display accommodation options"""
        st.subheader("Accommodation Options")
        
        if not accommodations:
            st.warning("No accommodation information available.")
            return
        
        df = pd.DataFrame(accommodations)
        st.dataframe(df, hide_index=True, use_container_width=True)
    
    def _display_dining(self, dining: List[Dict]):
        """Display dining recommendations"""
        st.subheader("Dining Recommendations")
        
        if not dining:
            st.warning("No dining information available.")
            return
        
        for i, restaurant in enumerate(dining, 1):
            with st.container():
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>{i}. {restaurant.get('Name', '')}</h4>
                    <p><strong>Cuisine:</strong> {restaurant.get('Cuisine', 'Unknown')}</p>
                    <p><strong>Price Range:</strong> {restaurant.get('Price Range', 'Unknown')}</p>
                    <p><strong>Rating:</strong> {restaurant.get('Rating', 'Unknown')}</p>
                    <p><strong>Specialty:</strong> {restaurant.get('Specialty', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def _display_local_tips(self, tips: str):
        """Display local tips"""
        if not tips:
            return
            
        st.subheader("Local Tips")
        st.markdown(tips)
    
    def _display_packing_list(self, packing: str):
        """Display packing list"""
        if not packing or not config["features"]["enable_packing"]:
            return
            
        st.subheader("Packing Suggestions")
        st.markdown(packing)
    
    def _save_trip(self, source: str, destination: str, data: Dict):
        """Save trip to session state"""
        trip = {
            "id": f"{source}-{destination}-{datetime.now().timestamp()}",
            "source": source,
            "destination": destination,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "data": data
        }
        
        st.session_state.saved_trips.append(trip)
        st.success("Trip saved successfully!")
    
    def _display_saved_trips(self):
        """Display saved trips in sidebar"""
        if not st.session_state.saved_trips:
            st.sidebar.write("No saved trips yet.")
            return
            
        st.sidebar.subheader("Saved Trips")
        
        for trip in st.session_state.saved_trips:
            with st.sidebar.expander(f"{trip['source']} → {trip['destination']}"):
                st.write(f"Date: {trip['date']}")
                if st.button("Load", key=f"load_{trip['id']}"):
                    st.session_state.travel_data = trip['data']
                    st.session_state.destination_info = trip['data']
                    st.experimental_rerun()
                
                if st.button("Delete", key=f"delete_{trip['id']}"):
                    st.session_state.saved_trips = [
                        t for t in st.session_state.saved_trips 
                        if t['id'] != trip['id']
                    ]
                    st.experimental_rerun()
    
    def _display_budget_calculator(self):
        """Display budget calculator"""
        if not config["features"]["enable_budget"]:
            return
            
        with st.expander("💰 Trip Budget Calculator"):
            st.write("Estimate your total trip cost based on selected options.")
            
            if not st.session_state.travel_data or not st.session_state.travel_data.get('travel_options'):
                st.warning("Generate travel options first to use the budget calculator.")
                return
                
            travel_options = st.session_state.travel_data['travel_options']
            accommodations = st.session_state.destination_info['accommodations'] if st.session_state.destination_info else []
            dining_options = st.session_state.destination_info['dining'] if st.session_state.destination_info else []
            
            # Travel selection
            st.subheader("Transportation")
            transport_mode = st.selectbox(
                "Select your transportation",
                [opt['Mode'] for opt in travel_options]
            )
            
            # Accommodation selection
            st.subheader("Accommodation")
            if accommodations:
                accommodation = st.selectbox(
                    "Select your accommodation",
                    [acc['Name'] for acc in accommodations]
                )
                nights = st.number_input("Number of nights", min_value=1, value=3)
            else:
                st.warning("No accommodation data available.")
                accommodation = None
                nights = 0
            
            # Dining selection
            st.subheader("Dining")
            if dining_options:
                meals_per_day = st.number_input("Meals per day", min_value=1, max_value=10, value=3)
                avg_meal_cost = st.number_input("Average meal cost (USD)", min_value=0, value=15)
            else:
                st.warning("No dining data available.")
                meals_per_day = 0
                avg_meal_cost = 0
            
            # Activities
            st.subheader("Activities")
            activities_cost = st.number_input("Estimated activities cost (USD)", min_value=0, value=100)
            
            # Calculate total
            if st.button("Calculate Budget"):
                # Get transport cost
                transport_cost = next(
                    (float(re.search(r'(\d+)', opt['Price (USD)']).group()) 
                    for opt in travel_options 
                    if opt['Mode'] == transport_mode)
                )
                
                # Get accommodation cost
                if accommodation:
                    acc_cost = next(
                        (float(re.search(r'(\d+)', acc['Price Range']).group()) 
                        for acc in accommodations 
                        if acc['Name'] == accommodation)
                    )
                    acc_total = acc_cost * nights
                else:
                    acc_total = 0
                
                # Calculate dining total
                dining_total = meals_per_day * avg_meal_cost * nights
                
                # Calculate total
                total = transport_cost + acc_total + dining_total + activities_cost
                
                # Display results
                st.success(f"Estimated Total Budget: ${total:,.2f}")
                
                # Breakdown
                st.write("### Budget Breakdown")
                breakdown = {
                    "Transportation": transport_cost,
                    "Accommodation": acc_total,
                    "Dining": dining_total,
                    "Activities": activities_cost
                }
                
                fig = px.pie(
                    names=list(breakdown.keys()),
                    values=list(breakdown.values()),
                    title="Budget Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    
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
            
            # Display saved trips if any
            if selected == "Saved Trips":
                self._display_saved_trips()
                return
            elif selected == "About":
                self._display_about()
                return
        
        # Main content
        st.title("✈️ AI Travel Planner Pro")
        
        # Tab layout
        tab1, tab2, tab3 = st.tabs(["Plan Your Trip", "Destination Info", "Trip Tools"])
        
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
                                # Get weather data
                                weather_data = self._get_weather(destination)
                                st.session_state.weather_data = weather_data
                                
                                # Generate travel recommendations
                                prompt = self._generate_travel_prompt(source, destination)
                                chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(prompt))
                                response = chain.run({})
                                
                                # Parse response
                                parsed_data = self._parse_travel_response(response)
                                st.session_state.travel_data = parsed_data
                                st.session_state.destination_info = parsed_data
                                
                                # Generate packing list
                                if config["features"]["enable_packing"]:
                                    packing_list = self._generate_packing_list(destination, weather_data)
                                    st.session_state.destination_info['packing'] = packing_list
                                
                                st.success("Travel plan generated successfully!")
                            except Exception as e:
                                st.error(f"Error generating travel plan: {e}")
            
            # Display results if available
            if st.session_state.travel_data:
                self._display_travel_options(st.session_state.travel_data['travel_options'])
                
                # Save trip button
                if config["features"]["enable_saving"]:
                    if st.button("💾 Save This Trip"):
                        self._save_trip(
                            source,
                            destination,
                            st.session_state.travel_data
                        )
        
        with tab2:
            if st.session_state.destination_info:
                # Destination overview
                st.header(f"🌍 {destination} Overview")
                
                # Weather display
                if config["features"]["enable_weather"]:
                    st.subheader("Current Weather")
                    if st.session_state.weather_data:
                        self._display_weather(st.session_state.weather_data)
                    else:
                        st.warning("Weather data not available.")
                
                # Map display
                if destination and config["features"]["enable_maps"]:
                    st.subheader("Destination Map")
                    self._display_destination_map(destination)
                
                # Display destination info
                self._display_attractions(st.session_state.destination_info['attractions'])
                self._display_accommodations(st.session_state.destination_info['accommodations'])
                self._display_dining(st.session_state.destination_info['dining'])
                self._display_local_tips(st.session_state.destination_info['tips'])
                self._display_packing_list(st.session_state.destination_info['packing'])
            else:
                st.info("Generate a travel plan first to see destination information.")
        
        with tab3:
            st.header("Trip Tools")
            
            # Budget calculator
            self._display_budget_calculator()
            
            # Packing list generator
            if config["features"]["enable_packing"]:
                with st.expander("🧳 Packing List Generator"):
                    if st.session_state.destination_info and st.session_state.destination_info.get('packing'):
                        st.markdown(st.session_state.destination_info['packing'])
                    else:
                        st.warning("Generate a travel plan first to get packing suggestions.")
            
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
        
        st.sidebar.subheader("Features")
        st.sidebar.write("""
        - Multi-modal travel options
        - Destination information
        - Accommodation recommendations
        - Dining suggestions
        - Packing lists
        - Budget calculator
        """)
        
        st.sidebar.subheader("Tech Stack")
        st.sidebar.write("""
        - Python
        - Streamlit
        - LangChain
        - Google Gemini
        - Plotly
        - Folium
        """)

# Run the application
if __name__ == "__main__":
    app = TravelPlanner()
    app.run()
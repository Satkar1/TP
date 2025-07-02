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
from langchain.memory import ConversationBufferMemory
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
            self.model_name = "gemini-2.0-flash-exp"
                
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.gemini_api_key,
                temperature=config["api"]["gemini"]["temperature"],
                max_output_tokens=config["api"]["gemini"]["max_tokens"]
            )
            
            # Setup conversation memory
            self.memory = ConversationBufferMemory()
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
            .chat-message {{
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                background-color: #f0f2f6;
            }}
            .user-message {{
                background-color: {primary_color};
                color: white;
                margin-left: 20%;
            }}
            .ai-message {{
                background-color: {secondary_color};
                color: white;
                margin-right: 20%;
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
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'current_source' not in st.session_state:
            st.session_state.current_source = ""
        if 'current_destination' not in st.session_state:
            st.session_state.current_destination = ""
    
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
        # Include chat history in prompt for context
        chat_context = ""
        if st.session_state.chat_history:
            chat_context = "\nPrevious conversation:\n" + "\n".join(
                f"{'User' if msg['role'] == 'user' else 'AI'}: {msg['content']}" 
                for msg in st.session_state.chat_history[-3:]  # Last 3 messages
            )
        
        return f"""
        You are an expert travel planner assistant. Provide comprehensive travel options from {source} to {destination}. 
        {chat_context}
        
        ## Travel Options
        
        Create a detailed comparison table with the following EXACT column headers:
        
        | Mode | Price Range (USD) | Duration | Comfort (1-5) | Directness | Notes |
        |------|------------------|----------|--------------|------------|-------|
        
        Fill ALL fields with realistic estimates. MUST include:
        - Flight
        - Train
        - Bus
        - Rental Car
        - Taxi/Uber
        
        Price ranges should be in USD (e.g., "$100-$200")
        Duration should be in hours/minutes (e.g., "2h 30m")
        Comfort should be 1-5 (5 being most comfortable)
        
        ## Destination Information
        
        [Rest of your prompt remains the same...]
        """
    
    def _generate_packing_list(self, destination: str, weather_data: Optional[Dict] = None) -> str:
        """Generate packing list based on destination and weather"""
        weather_desc = weather_data['weather'][0]['description'] if weather_data else 'unknown'
        temp = weather_data['main']['temp'] if weather_data else 'unknown'
        
        prompt = f"""
        Create a detailed packing list for a trip to {destination}. Consider:
        - Current weather: {weather_desc}
        - Temperature: {temp}°C
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
            result = chain.run({})
            
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Generated packing list for {destination}"
            })
            
            return result
        except Exception as e:
            return f"Could not generate packing list: {e}"
    
    def _parse_travel_response(self, response: str) -> Dict:
        """Parse the LLM response into structured data with improved error handling"""
        result = {
            "travel_options": [],
            "attractions": [],
            "accommodations": [],
            "dining": [],
            "tips": "",
            "packing": ""
        }
        
        # Debug: Show raw response
        if config.get("debug", False):
            with st.expander("Debug: Raw Response"):
                st.code(response)
        
        # Parse travel options with more flexible matching
        try:
            travel_section = re.search(r"## Travel Options([\s\S]+?)(?=##|\Z)", response)
            if travel_section:
                travel_content = travel_section.group(1)
                table_match = re.search(r"\|.+\|.+\|\n\|[-:]+\|[-:]+\|([\s\S]+?)(?=\n\n|\Z)", travel_content)
                if table_match:
                    result["travel_options"] = self._parse_markdown_table(table_match.group(0))
        except Exception as e:
            st.warning(f"Error parsing travel options: {e}")
        
        # Parse other sections (attractions, accommodations, etc.)
        # [Rest of your parsing code remains the same...]
        
        return result
    
    def _parse_markdown_table(self, table_text: str) -> List[Dict]:
        """Parse markdown table into list of dictionaries with improved handling"""
        if not table_text:
            return []
            
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        if len(lines) < 2:
            return []
        
        # Extract headers from the first line
        headers = [h.strip() for h in lines[0].split('|')[1:-1]]
        data = []
        
        for line in lines[2:]:
            values = [v.strip() for v in line.split('|')[1:-1]]
            if len(values) == len(headers):
                data.append(dict(zip(headers, values)))
        
        return data
    
    def _display_travel_options(self, options: List[Dict]):
        """Display travel options with improved validation and visualization"""
        st.subheader("Travel Options")
        
        if not options:
            st.error("No travel options could be generated. Please try again with different cities.")
            return
        
        try:
            # Convert to DataFrame with validation
            df = pd.DataFrame(options)
            
            # Check for required columns and fill missing ones
            required_cols = ['Mode', 'Price Range (USD)', 'Duration']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = "Not available"
            
            # Clean and process data
            try:
                df['Price Min'] = df['Price Range (USD)'].str.extract(r'(\d+)').astype(float)
                df['Price Max'] = df['Price Range (USD)'].str.extract(r'(\d+)[^\d]*$').astype(float)
                df['Price Avg'] = (df['Price Min'] + df['Price Max']) / 2
            except:
                df['Price Min'] = 0
                df['Price Max'] = 0
                df['Price Avg'] = 0
            
            # Extract numeric comfort if available
            if 'Comfort (1-5)' in df.columns:
                try:
                    df['Comfort'] = df['Comfort (1-5)'].str.extract(r'(\d)').astype(float)
                except:
                    df['Comfort'] = 3  # Default value
            
            # Display
            col1, col2 = st.columns(2)
            
            with col1:
                display_cols = [c for c in df.columns if c not in ['Price Min', 'Price Max', 'Price Avg', 'Comfort']]
                st.dataframe(df[display_cols], 
                            hide_index=True, 
                            use_container_width=True)
            
            with col2:
                # Price comparison if we have valid prices
                if 'Price Avg' in df.columns and df['Price Avg'].notna().any():
                    fig_price = px.bar(
                        df,
                        x='Mode',
                        y='Price Avg',
                        error_y=df['Price Max'] - df['Price Avg'],
                        error_y_minus=df['Price Avg'] - df['Price Min'],
                        title='Price Comparison (USD)',
                        labels={'Price Avg': 'Average Price'},
                        color='Mode'
                    )
                    st.plotly_chart(fig_price, use_container_width=True)
                
                # Comfort comparison if available
                if 'Comfort' in df.columns and df['Comfort'].notna().any():
                    fig_comfort = px.bar(
                        df,
                        x='Mode',
                        y='Comfort',
                        title='Comfort Level (1-5)',
                        color='Mode',
                        range_y=[0, 5]
                    )
                    st.plotly_chart(fig_comfort, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error displaying travel options: {str(e)}")
            if config.get("debug", False):
                st.write("Options data:", options)

    def _display_destination_map(self, destination: str):
        """Display a map centered on the destination"""
        if not config["features"]["enable_maps"]:
            return
            
        try:
            # Simplified map - in production would use geocoding API
            m = folium.Map(location=[20, 0], zoom_start=2)
            folium.Marker(
                location=[20, 0],  # Placeholder coordinates
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
        
        # Add to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": f"Saved trip from {source} to {destination}"
        })
    
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
                    st.session_state.current_source = trip['source']
                    st.session_state.current_destination = trip['destination']
                    st.experimental_rerun()
                
                if st.button("Delete", key=f"delete_{trip['id']}"):
                    st.session_state.saved_trips = [
                        t for t in st.session_state.saved_trips 
                        if t['id'] != trip['id']
                    ]
                    st.experimental_rerun()
    
    def _display_budget_calculator(self):
        """Display budget calculator with improved error handling"""
        if not config["features"]["enable_budget"]:
            return
            
        with st.expander("💰 Trip Budget Calculator"):
            st.write("Estimate your total trip cost based on selected options.")
            
            if not st.session_state.travel_data or not st.session_state.travel_data.get('travel_options'):
                st.warning("Generate travel options first to use the budget calculator.")
                return
                
            travel_options = st.session_state.travel_data['travel_options']
            accommodations = st.session_state.destination_info.get('accommodations', []) if st.session_state.destination_info else []
            dining_options = st.session_state.destination_info.get('dining', []) if st.session_state.destination_info else []
            
            # Travel selection
            st.subheader("Transportation")
            try:
                transport_modes = [opt.get('Mode', 'Unknown') for opt in travel_options]
                transport_mode = st.selectbox(
                    "Select your transportation",
                    transport_modes
                )
            except:
                st.warning("No valid transportation options available.")
                return
            
            # Accommodation selection
            st.subheader("Accommodation")
            if accommodations:
                try:
                    accommodation_names = [acc.get('Name', 'Unknown') for acc in accommodations]
                    accommodation = st.selectbox(
                        "Select your accommodation",
                        accommodation_names
                    )
                    nights = st.number_input("Number of nights", min_value=1, value=3)
                except:
                    st.warning("No valid accommodation data available.")
                    accommodation = None
                    nights = 0
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
                # Initialize totals
                transport_cost = 0
                acc_total = 0
                dining_total = 0
                
                # Get transport cost
                try:
                    for opt in travel_options:
                        if opt.get('Mode') == transport_mode:
                            price_str = opt.get('Price Range (USD)', '$0')
                            prices = re.findall(r'\$(\d+)', price_str)
                            if prices:
                                transport_cost = float(prices[0])  # Use first price found
                            break
                except:
                    transport_cost = 0
                
                # Get accommodation cost
                if accommodation:
                    try:
                        for acc in accommodations:
                            if acc.get('Name') == accommodation:
                                price_str = acc.get('Price Range', '$0')
                                prices = re.findall(r'\$(\d+)', price_str)
                                if prices:
                                    acc_cost = float(prices[0])  # Use first price found
                                    acc_total = acc_cost * nights
                                break
                    except:
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
    
    def _display_chat(self):
        """Display chat interface"""
        st.subheader("Travel Assistant Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            role = "user" if message["role"] == "user" else "assistant"
            st.markdown(f"""
            <div class="chat-message {'user-message' if role == 'user' else 'ai-message'}">
                <strong>{'You' if role == 'user' else 'Travel Assistant'}:</strong>
                <p>{message['content']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input("Ask a travel question:", key="chat_input")
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Generate response
            with st.spinner("Thinking..."):
                try:
                    prompt = f"""
                    You are a travel assistant. Continue this conversation based on the context:
                    
                    Current trip: From {st.session_state.get('current_source', 'unknown')} to {st.session_state.get('current_destination', 'unknown')}
                    
                    Previous conversation:
                    {''.join([f"{m['role']}: {m['content']}\n" for m in st.session_state.chat_history[-3:]])}
                    
                    Current question: {user_input}
                    
                    Provide a helpful, concise response.
                    """
                    
                    chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(prompt))
                    response = chain.run({})
                    
                    # Add AI response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Rerun to show new messages
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    
    def run(self):
        """Run the main application"""
        # Sidebar
        with st.sidebar:
            st.title(config["app"]["name"])
            st.write(config["app"]["description"])
            
            # Navigation
            selected = option_menu(
                menu_title=None,
                options=["Plan Trip", "Saved Trips", "Chat", "About"],
                icons=["compass", "bookmark", "chat", "info-circle"],
                default_index=0
            )
            
            # Display appropriate section based on selection
            if selected == "Saved Trips":
                self._display_saved_trips()
                return
            elif selected == "About":
                self._display_about()
                return
            elif selected == "Chat":
                self._display_chat()
                return
        
        # Main content
        st.title("✈️ AI Travel Planner Pro")
        
        # Tab layout
        tab1, tab2, tab3 = st.tabs(["Plan Your Trip", "Destination Info", "Trip Tools"])
        
        with tab1:
            with st.form("travel_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    source = st.text_input("Departure City", 
                                         placeholder="New York",
                                         value=st.session_state.get('current_source', ''))
                
                with col2:
                    destination = st.text_input("Destination City", 
                                              placeholder="Paris",
                                              value=st.session_state.get('current_destination', ''))
                
                submitted = st.form_submit_button("Generate Travel Plan")
                
                if submitted:
                    if not source or not destination:
                        st.error("Please enter both departure and destination cities.")
                    else:
                        with st.spinner("Generating travel recommendations..."):
                            try:
                                # Store current locations
                                st.session_state.current_source = source
                                st.session_state.current_destination = destination
                                
                                # Get weather data
                                weather_data = self._get_weather(destination)
                                st.session_state.weather_data = weather_data
                                
                                # Generate travel recommendations
                                prompt = self._generate_travel_prompt(source, destination)
                                chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(prompt))
                                response = chain.run({})
                                
                                # Add to chat history
                                st.session_state.chat_history.append({
                                    "role": "user",
                                    "content": f"Requested travel plan from {source} to {destination}"
                                })
                                
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
                self._display_travel_options(st.session_state.travel_data.get('travel_options', []))
                
                # Save trip button
                if config["features"]["enable_saving"]:
                    if st.button("💾 Save This Trip"):
                        self._save_trip(
                            st.session_state.current_source,
                            st.session_state.current_destination,
                            st.session_state.travel_data
                        )
        
        with tab2:
            if st.session_state.destination_info:
                # Destination overview
                st.header(f"🌍 {st.session_state.current_destination} Overview")
                
                # Weather display
                if config["features"]["enable_weather"]:
                    st.subheader("Current Weather")
                    if st.session_state.weather_data:
                        self._display_weather(st.session_state.weather_data)
                    else:
                        st.warning("Weather data not available.")
                
                # Map display
                if st.session_state.current_destination and config["features"]["enable_maps"]:
                    st.subheader("Destination Map")
                    self._display_destination_map(st.session_state.current_destination)
                
                # Display destination info
                self._display_attractions(st.session_state.destination_info.get('attractions', []))
                self._display_accommodations(st.session_state.destination_info.get('accommodations', []))
                self._display_dining(st.session_state.destination_info.get('dining', []))
                self._display_local_tips(st.session_state.destination_info.get('tips', ''))
                self._display_packing_list(st.session_state.destination_info.get('packing', ''))
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
        - Chat assistant
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

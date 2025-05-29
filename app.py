import streamlit as st
import json
import tempfile
import os
import time
import cv2
import numpy as np
import pandas as pd
from video_processing import process_video
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from ollama_chat import query_ollama


# Set wide layout to ensure canvas displays fully
st.set_page_config(layout="wide")


# ============================================================================
# ZONE DRAWING FUNCTION
# ============================================================================
def draw_zones(video_path):
    """
    Allows users to draw entry and exit zones on the first frame of the video using Streamlit canvas.
    """
    st.markdown('<div class="section-title">Draw Points for Zones</div>', unsafe_allow_html=True)
    with st.expander("Instructions", expanded=False):#expanded=False means it starts closed
        st.markdown("""
        **Instructions for Drawing Points:**
        - Draw a polygon by clicking to add points:
        - Left-click to add a point.
        - Right-click to close the polygon.
        - Double-click to remove the last point.
        """)
    
    # Load the first frame of the video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if not ret:
        st.error("Could not read video frame")
        return
        
    # Calculate appropriate display size to fit in browser
    max_display_width = 1000 
    max_display_height = 600  
    
    # Calculate scaling to fit within display limits while maintaining aspect ratio
    width_scale = max_display_width / original_width
    height_scale = max_display_height / original_height
    display_scale = min(width_scale, height_scale, 1.0)  
    
    display_width = int(original_width * display_scale)
    display_height = int(original_height * display_scale)
    
    # Resize frame for display
    if display_scale < 1.0:
        display_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
    else:
        display_frame = frame.copy()
    
    # Convert to RGB for PIL
    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    
    # Initialize zones in session state
    if "zones" not in st.session_state:
        st.session_state.zones = {"entry_zones": {}, "exit_zones": {}}
    
    # Create background image with existing zones overlay
    background_image = frame_pil.copy()
    if st.session_state.zones["entry_zones"] or st.session_state.zones["exit_zones"]:
        draw = ImageDraw.Draw(background_image)
        
        # Draw existing entry zones in yellow
        for zone_name, polygon in st.session_state.zones["entry_zones"].items():
            coords = [(x * display_scale, y * display_scale) for x, y in polygon.exterior.coords[:-1]]
            if len(coords) > 2:
                draw.polygon(coords, outline="yellow", width=3)
                centroid = polygon.centroid
                draw.text((centroid.x * display_scale, centroid.y * display_scale), 
                         zone_name, fill="yellow")
        
        # Draw existing exit zones in magenta
        for zone_name, polygon in st.session_state.zones["exit_zones"].items():
            coords = [(x * display_scale, y * display_scale) for x, y in polygon.exterior.coords[:-1]]
            if len(coords) > 2:
                draw.polygon(coords, outline="magenta", width=3)
                centroid = polygon.centroid
                draw.text((centroid.x * display_scale, centroid.y * display_scale), 
                         zone_name, fill="magenta")
    
    #  zone drawing interface
    st.write("**Draw Polygon Zones**")

    # Ensure the form block is properly defined
    with st.form(key="poly_zone_form"):
        col1, col2 = st.columns(2)
        with col1:
            zone_type_poly = st.selectbox("Zone Type", ["Entry", "Exit"], key="zone_type_poly")
        with col2:
            zone_name_poly = st.text_input("Zone Name (e.g., North_in, South_out)", key="zone_name_poly")
        
        st.write(f"Draw a polygon zone on the image below (Canvas size: {display_width} × {display_height}):")
        
        if "canvas_key" not in st.session_state:
            st.session_state.canvas_key = "canvas_poly_0"
        
        canvas_result_poly = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)" if zone_type_poly == "Entry" else "rgba(255, 0, 255, 0.3)",
            stroke_width=3,
            stroke_color="#FFFF00" if zone_type_poly == "Entry" else "#FF00FF",
            background_image=background_image,
            height=display_height,
            width=display_width,
            drawing_mode="polygon",
            key=st.session_state.canvas_key,
            display_toolbar=True,
            update_streamlit=True
        )
        
        submit_poly_zone = st.form_submit_button("Save Polygon Zone")

    # Process the drawn polygon
    if submit_poly_zone and zone_name_poly and canvas_result_poly:
        if zone_name_poly in st.session_state.zones["entry_zones"] or zone_name_poly in st.session_state.zones["exit_zones"]:
            st.error(f"Zone name '{zone_name_poly}' already exists. Please use a different name.")
        else:
            st.write("Debug: Full canvas json_data:", canvas_result_poly.json_data)
            
            objects = canvas_result_poly.json_data.get("objects", []) if canvas_result_poly.json_data else []
            saved = False
            
            st.write("Debug: Raw canvas objects:", objects)
            
            for obj in objects:
                points = []
                seen_points = set()
                first_point = None
                
                if obj["type"] == "path":
                    path = obj.get("path", [])
                    for i, cmd in enumerate(path):
                        if cmd[0] in ["M", "L"]:
                            x, y = cmd[1], cmd[2]
                            # Store the first point for comparison
                            if i == 0:
                                first_point = (x, y)
                                point = (round(x, 2), round(y, 2))
                                seen_points.add(point)
                                orig_x = x / display_scale
                                orig_y = y / display_scale
                                points.append((orig_x, orig_y))
                            else:
                                # Check if this point is the closing point (matches the first point)
                                distance = ((x - first_point[0]) ** 2 + (y - first_point[1]) ** 2) ** 0.5
                                if i == len(path) - 1 and distance < 1.0:  # Tolerance of 1 pixel
                                    continue  # Skip the closing point
                                point = (round(x, 2), round(y, 2))
                                if point not in seen_points:
                                    seen_points.add(point)
                                    orig_x = x / display_scale
                                    orig_y = y / display_scale
                                    points.append((orig_x, orig_y))
                
                st.write(f"Debug: Object type: {obj.get('type')}, Extracted points (original scale):", points)
                
                if len(points) >= 3:
                    zone_key = "entry_zones" if zone_type_poly == "Entry" else "exit_zones"
                    try:
                        st.session_state.zones[zone_key][zone_name_poly] = Polygon(points)
                        st.success(f"Saved {zone_type_poly.lower()} zone: **{zone_name_poly}**")
                        st.info(f"Zone coordinates (original scale): {[(int(x), int(y)) for x, y in points]}")
                        saved = True
                        st.session_state.canvas_key = f"canvas_poly_{len(st.session_state.zones['entry_zones']) + len(st.session_state.zones['exit_zones'])}"
                        st.rerun()
                        break
                    except Exception as e:
                        st.error(f"Error creating polygon: {str(e)}")
                else:
                    st.warning(f"Not enough points to form a polygon. Found {len(points)} points, need at least 3.")
            
            if not saved:
                if objects:
                    st.warning("No valid polygon detected. Ensure you closed the polygon by right-clicking.")
                else:
                    st.warning("No shapes detected. Please draw a polygon on the canvas and close it by right-clicking.")

    
    # Zone management 
    st.write("---")
    st.subheader("Zone Management")
    
    # Display current zones
    total_zones = len(st.session_state.zones["entry_zones"]) + len(st.session_state.zones["exit_zones"])
    
    if total_zones > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.zones["entry_zones"]:
                st.write("**Entry Zones:**")
                for zone_name in st.session_state.zones["entry_zones"].keys():
                    st.write(f"• {zone_name}")
        
        with col2:
            if st.session_state.zones["exit_zones"]:
                st.write("**Exit Zones:**")
                for zone_name in st.session_state.zones["exit_zones"].keys():
                    st.write(f"• {zone_name}")
        
        # Zone deletion
        all_zones = list(st.session_state.zones["entry_zones"].keys()) + list(st.session_state.zones["exit_zones"].keys())
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            zone_to_delete = st.selectbox("Select zone to delete:", [""] + all_zones, key="zone_delete_select")
        
        with col2:
            if st.button("Delete Zone") and zone_to_delete:
                if zone_to_delete in st.session_state.zones["entry_zones"]:
                    del st.session_state.zones["entry_zones"][zone_to_delete]
                elif zone_to_delete in st.session_state.zones["exit_zones"]:
                    del st.session_state.zones["exit_zones"][zone_to_delete]
                st.success(f"Deleted zone: {zone_to_delete}")
                st.rerun()
                
        
        with col3:
            if st.button("Clear All") and total_zones > 0:
                st.session_state.zones = {"entry_zones": {}, "exit_zones": {}}
                st.success("All zones cleared")
                st.rerun()
    else:
        st.info("No zones defined yet. Draw some zones above to get started.")
    
    # Save zones to file
    if total_zones > 0:
        st.write("---")
        if st.button("Save All Zones to File", type="primary"):
            try:
                required_vars = {
                    "original_width": original_width,
                    "original_height": original_height,
                    "display_scale": display_scale,
                    "total_zones": total_zones
                }
            except NameError as e:
                st.error(f"Error: Missing required variable - {str(e)}. Ensure original_width, original_height, display_scale, and total_zones are defined.")
                st.stop()

            zones_data = {
                "entry_zones": {},
                "exit_zones": {},
                "metadata": {
                    "original_width": original_width,
                    "original_height": original_height,
                    "display_scale": display_scale,
                    "total_zones": total_zones
                }
            }

            # Convert entry_zones to raw coordinates
            for zone_name, zone_data in st.session_state.zones["entry_zones"].items():
                if isinstance(zone_data, Polygon):
                    points = list(zone_data.exterior.coords)[:-1]
                    print(f"Debug: Saving entry zone {zone_name} with {len(points)} points: {points}")
                else:
                    points = zone_data
                zones_data["entry_zones"][zone_name] = [[round(x, 2), round(y, 2)] for x, y in points]

            # Convert exit_zones to raw coordinates
            for zone_name, zone_data in st.session_state.zones["exit_zones"].items():
                if isinstance(zone_data, Polygon):
                    points = list(zone_data.exterior.coords)[:-1]
                    print(f"Debug: Saving exit zone {zone_name} with {len(points)} points: {points}")
                else:
                    points = zone_data
                zones_data["exit_zones"][zone_name] = [[round(x, 2), round(y, 2)] for x, y in points]

            try:
                zones_file_path = os.path.abspath("zones.json")
                st.write(f"Debug: Saving to {zones_file_path}")
                with open(zones_file_path, "w") as f:
                    json.dump(zones_data, f, indent=2)
                st.success("Zones saved to zones.json")

                zones_json = json.dumps(zones_data, indent=2)
                st.download_button(
                    label="Download zones.json",
                    data=zones_json,
                    file_name="zones.json",
                    mime="application/json"
                )
            except PermissionError as e:
                st.error(f"Permission error: Cannot write to {zones_file_path}. Ensure the app has write permissions and the file is not open in another program. Details: {str(e)}")
            except Exception as e:
                st.error(f"Error saving zones: {str(e)}")       

# ============================================================================
# CONVERSATIONAL AI
# ============================================================================
def chat_interface():
    """
    Implements a chatbot interface for video analytics Q&A.

    Algorithm :
    - Retrieval-Augmented Generation (RAG): LLM answers are grounded in the structured analytics data as it stored in FAISS DB.
    - Local LLM (Ollama): Ensures data privacy, low latency, and no external API costs.
    - Session state maintains chat history and context for a seamless conversation.
    - System prompt strictly bounds the assistant to only answer from analytics data (no hallucination).   
    """
    st.title("Car Turn Detection Analytics Chatbot")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are an expert assistant for vehicle turn tracking analytics and traffic pattern analysis. You have access to structured video analytics data containing all cars detected, their unique IDs, the turns they made (right turn, left turn, U-turn, or none). You provide complete analysis including total vehicle counts and detailed turn statistics. Answer questions accurately and concisely based only on this analytics data including individual car behavior, turn patterns, and traffic insights. If you don't know the answer or if the answer is not present in the analytics, say so. Do not make anything up if you haven't been provided with relevant context."}
        ]
    
    # Retrieve analytics data in order to serve as the knowledge base for RAG.
    analytics = st.session_state.get("analytics", {})
    # User input for questions about analytics
    user_question = st.text_input("Ask a question about the turn detection analytics:")
    
    if user_question:
        # Add user message to conversation history
        st.session_state.messages.append({"role": "user", "content": user_question})
        # Query the LLM with analytics context and user question
        response = query_ollama(analytics, user_question)
        # Display the last user and assistant messages for context
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display the last user and assistant messages for context
    if len(st.session_state.messages) >= 2:
        user_msg = st.session_state.messages[-2]
        assistant_msg = st.session_state.messages[-1]
        
        if user_msg["role"] == "user":
            st.write(f"**You:** {user_msg['content']}")
        if assistant_msg["role"] == "assistant":
            st.write(f"**Assistant:** {assistant_msg['content']}")
            
# ============================================================================
# VIDEO PROCESSING AND STORAGE FUNCTIONS
# ============================================================================

def process_and_store_video(video_path, zones):
    """
    Handles uploaded video file:
    - Stores it temporarily for processing.
    - Calls the detection algorithms.
    - Stores results in Streamlit's session state.
    - Path to the input video file.
    - Dictionary containing entry_zones and exit_zones (with Polygon objects or coordinates).

    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Generate output path for processed video
    output_path = video_path.rsplit('.', 1)[0] + '_analyzed.mp4'
    
    # Pass zones directly to process_video
    output_video, analytics = process_video(video_path, output_path, zones_file=zones)
    
    # Store results in Streamlit's session state
    st.session_state.output_video = output_video
    st.session_state.analytics = analytics
    
    return output_video, analytics

# ============================================================================
# MAIN APPLICATION INTERFACE
# ============================================================================
def main():
    """
    Main application workflow:
    - Step 1: Upload a video
    - Step 2: Draw zones
    - Step 3: Process and analyze the video (YOLO + DeepSORT + Turn Detection)
    - Step 4: View analytics and interact with chatbot (RAG-based)
    """
    st.title("Car Turn Detection")

    # Initialize session state to manage navigation and data persistence
    if "step" not in st.session_state:
        st.session_state.step = "upload"
    if "zones" not in st.session_state:
        # Stores manually drawn polygon zones (entry/exit)
        st.session_state.zones = {"entry_zones": {}, "exit_zones": {}}
    if "temp_video_path" not in st.session_state:
        st.session_state.temp_video_path = None
    if "output_video" not in st.session_state:
        st.session_state.output_video = None
    if "analytics" not in st.session_state:
        st.session_state.analytics = None
    if "video_uploaded" not in st.session_state:
        st.session_state.video_uploaded = False
    
    # Step 1: Upload Video
    if st.session_state.step == "upload":
        st.header("Step 1: Upload Video")
        uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "mov", "avi"])
        
        if uploaded_file:
            st.success("Video uploaded successfully!")
            # Store uploaded video to a temporary location for further processing
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, f"uploaded_video_{int(time.time())}.mp4")
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            st.session_state.temp_video_path = temp_path
            st.session_state.video_uploaded = True
            
            # Preview the uploaded video
            st.video(uploaded_file)
            
            if st.button("Proceed to Draw Zones"):
                st.session_state.step = "draw_zones"
                st.rerun()

    # Step 2: Draw Zones
    elif st.session_state.step == "draw_zones":
        st.header("Step 2: Draw Zones")
        
        # Check if video still exists
        if not st.session_state.temp_video_path or not os.path.exists(st.session_state.temp_video_path):
            st.error("Video file not found. Please go back and upload a video again.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Back to Upload"):
                    # Reset state if video is missing.
                    st.session_state.step = "upload"
                    st.session_state.video_uploaded = False
                    st.session_state.temp_video_path = None
                    st.session_state.zones = {"entry_zones": {}, "exit_zones": {}}
                    st.rerun()

            st.stop()
        
        # Call the draw_zones function
        draw_zones(st.session_state.temp_video_path)
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back to Upload"):
                # Clean up temporary video file and reset state
                if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
                    try:
                        os.unlink(st.session_state.temp_video_path)
                    except:
                        pass
                st.session_state.temp_video_path = None
                st.session_state.video_uploaded = False
                st.session_state.zones = {"entry_zones": {}, "exit_zones": {}}
                st.session_state.step = "upload"
                st.rerun()

        
        with col2:
            # Require at least one zone before proceeding to processing.
            total_zones = len(st.session_state.zones["entry_zones"]) + len(st.session_state.zones["exit_zones"])
            if total_zones == 0:
                # st.warning("Please define at least one zone before proceeding.")
                st.button("Proceed to Process", disabled=True)
            else:
                st.success(f"Total zones defined: {total_zones}")
                if st.button("Proceed to Process"):
                    st.session_state.step = "process"
                    st.rerun()


    # Step 3: Process Video and Show Analytics/Chatbot
    elif st.session_state.step == "process":
        st.header("Step 3: Process Video and View Analytics")
        
        # Verify video and zones
        if not st.session_state.temp_video_path or not os.path.exists(st.session_state.temp_video_path):
            st.error("Video file not found. Please go back and upload a video again.")
            if st.button("Back to Upload"):
                st.session_state.step = "upload"
                st.session_state.video_uploaded = False
                st.session_state.temp_video_path = None
                st.session_state.zones = {"entry_zones": {}, "exit_zones": {}}
                st.rerun()
            st.stop()
        
        # Display current video and zone summary
        total_zones = len(st.session_state.zones["entry_zones"]) + len(st.session_state.zones["exit_zones"])
        st.info(f"Video: {os.path.basename(st.session_state.temp_video_path)} | Zones: {total_zones}")
        
        # Show entry and exit zones for user reference
        if st.session_state.zones["entry_zones"]:
            st.write("**Entry Zones:**", list(st.session_state.zones["entry_zones"].keys()))
        if st.session_state.zones["exit_zones"]:
            st.write("**Exit Zones:**", list(st.session_state.zones["exit_zones"].keys()))
        
        # Process video if not already processed
        if st.session_state.output_video is None:
            if st.button("Process Video", type="primary"):
                with st.spinner("Processing your video... This may take several minutes."):
                    try:
                        # process_and_store_video(st.session_state.temp_video_path, st.session_state.zones)
                        process_and_store_video(st.session_state.temp_video_path, '/Users/kanishka/Desktop/car-turn-detection/zones.json')
                        st.success("Video processing completed successfully!")
                    except Exception as e:
                        st.error(f"Error during video processing: {str(e)}")
                        st.write("**Debug info:** Make sure YOLO models and dependencies are properly installed.")
                        import traceback
                        st.text(traceback.format_exc())


        # Display Results and Analytics 
        if st.session_state.output_video and st.session_state.analytics:
            st.subheader("Processing Results")
            
            # Analytics display
            with st.expander("View Analytics Data", expanded=True):
                st.json(st.session_state.analytics)
            
            # Download processed video
            if os.path.exists(st.session_state.output_video):
                with open(st.session_state.output_video, "rb") as f:
                    st.download_button(
                        label="Download Processed Video",
                        data=f.read(),
                        file_name="processed_output.mp4",
                        mime="video/mp4"
                    )
            
            # Chatbot and Report Generation
            # Integrate a RAG-based chatbot for interactive analytics Q&A.
            if "show_chat" not in st.session_state:
                st.session_state.show_chat = False
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Chat with AI about Results"):
                    st.session_state.show_chat = not st.session_state.show_chat
            
            with col2:
                if st.button("Generate Report"):
                    # Placeholder for future report generation feature.
                    st.info("Report generation feature can be implemented here")
            
            # Show chat interface if enabled
            if st.session_state.show_chat:
                st.markdown("---")
                chat_interface()
        
        # Navigation
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Back to Draw Zones"):
                # Allow user to re-define zones without re-uploading the video
                st.session_state.step = "draw_zones"
                st.session_state.output_video = None
                st.session_state.analytics = None
                st.rerun()

        
        with col2:
            if st.button("Process New Video"):
                # Clean up current session
                if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
                    try:
                        os.unlink(st.session_state.temp_video_path)
                    except:
                        pass
                if st.session_state.output_video and os.path.exists(st.session_state.output_video):
                    try:
                        os.unlink(st.session_state.output_video)
                    except:
                        pass
                
                # Reset session state
                st.session_state.step = "upload"
                st.session_state.temp_video_path = None
                st.session_state.video_uploaded = False
                st.session_state.output_video = None
                st.session_state.analytics = None
                st.session_state.zones = {"entry_zones": {}, "exit_zones": {}}
                st.rerun()

        
        with col3:
            if st.button("Save Session"):
                # Save current session state to file
                session_data = {
                    "zones": {
                        "entry_zones": {k: [[x, y] for x, y in v.exterior.coords[:-1]] for k, v in st.session_state.zones["entry_zones"].items()},
                        "exit_zones": {k: [[x, y] for x, y in v.exterior.coords[:-1]] for k, v in st.session_state.zones["exit_zones"].items()}
                    },
                    "analytics": st.session_state.analytics
                }
                
                session_file = f"session_{int(time.time())}.json"
                with open(session_file, "w") as f:
                    json.dump(session_data, f, indent=4)
                st.success(f"Session saved to {session_file}")

#Helper function to improve error handling
def safe_cleanup():
    """Safely clean up temporary files"""
    try:
        if hasattr(st.session_state, 'temp_video_path') and st.session_state.temp_video_path:
            if os.path.exists(st.session_state.temp_video_path):
                os.unlink(st.session_state.temp_video_path)
        
        if hasattr(st.session_state, 'output_video') and st.session_state.output_video:
            if os.path.exists(st.session_state.output_video):
                os.unlink(st.session_state.output_video)
    except Exception as e:
        st.sidebar.warning(f"Cleanup warning: {str(e)}")



if __name__ == "__main__":
    main()



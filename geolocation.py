import streamlit.components.v1 as components
import streamlit as st

def get_geolocation():
    loc_json = components.html(
        """
        <script>
        const sendLocation = (lat, lon) => {
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: JSON.stringify({lat: lat, lon: lon}),
            }, "*");
        }

        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => sendLocation(position.coords.latitude, position.coords.longitude),
                (error) => sendLocation(null, null)
            );
        } else {
            sendLocation(null, null);
        }
        </script>
        """,
        height=0,
        key="geolocation"
    )
    return loc_json
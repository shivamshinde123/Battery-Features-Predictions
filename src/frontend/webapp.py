
import streamlit as st
import numpy as np
import os
import pickle
from ..utils import Utility

logger = Utility().setup_logger()

class Webapp:

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.abspath(os.path.join(self.script_dir, '..', '..'))

    def app(self):

        """streamlit app"""

        try:
            st.title('State of Charge (SoC) Prediction')

            battery_voltage_v = st.number_input('Battery Voltage (V)')
            heating_power_can_kW = st.number_input('Heating Power CAN (kW)')
            heater_voltage_v = st.number_input('Heater Voltage (v)')
            coolant_volume_flow_500_l_h = st.number_input('Coolant Volume Flow (l/h)')
            battery_current_a = st.number_input('Battery Current (A)')

            input = np.array([battery_voltage_v, heating_power_can_kW, heater_voltage_v, coolant_volume_flow_500_l_h, battery_current_a])

            predict_soc_btn = st.button('Predict SoC')

            with open(os.path.join(self.root_dir, 'Models', 'xgb_model.pkl'), 'rb') as file:
                model = pickle.load(file)

            if predict_soc_btn:
                with st.spinner("Predicting SoC..."):

                    soc = model.predict(input.reshape(1, -1))

                    st.success(f"State of Charge (SoC): {soc[0]}.")

        except Exception as e:
            logger.error("Webapp loading failed.", exc_info=e)
            raise

if __name__ == "__main__":

    wa = Webapp()
    wa.app()
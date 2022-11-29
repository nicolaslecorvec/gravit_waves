import streamlit as st
import h5py


st.markdown("""
    # Detector of Continuous Gravitational Waves

    ## Please put your data file in hdf5 format

""")


uploaded_file = st.file_uploader("Choose a file",  type=['hdf5'])
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)











st.write('Your data chance to be a Continuous Gravitational Waves:', result)

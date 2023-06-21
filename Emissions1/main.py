import streamlit as st
from collections import OrderedDict

import configuration

from tabs import dataviz, intro, conclusion, modeling, user_selection,interpretability






st.markdown('---')
st.markdown(
    """
    <style>
    body {
        background-color: #F0F0F0; /* Cambia el color de fondo a gris claro (#F0F0F0) */
    }
    </style>
    """,
    unsafe_allow_html=True
)




TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (dataviz.sidebar_name, dataviz),
        (modeling.sidebar_name,modeling),
        (interpretability.sidebar_name,interpretability),
        (user_selection.sidebar_name, user_selection),
        (conclusion.sidebar_name,conclusion),
         
        
    ]
    
    
)


def run():
    st.sidebar.image(
        "CO2.png",
        width=200,
    )
    
    tab_name = st.sidebar.radio("", list(TABS.keys()),0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {configuration.PROMOTION}", unsafe_allow_html=True)
    
    
    for member in configuration.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)
    
    tab = TABS[tab_name]
    tab.run()
    
if __name__ == '__main__':
    run()

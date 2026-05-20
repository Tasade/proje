import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title='DataWizard Dashboard', page_icon='⚡', layout='wide')

@st.cache_data
def load_data():
    try:
        return pd.read_excel('data/cleaned_data.xlsx')
    except Exception:
        return pd.read_csv('data/cleaned_data.csv')

df = load_data()
st.sidebar.title('Filtreler')
st.sidebar.markdown(f'**Toplam Kayit:** {len(df):,}')
st.sidebar.divider()
    u0=sorted(df['OKUL ADI'].dropna().unique().tolist())
    s0=st.sidebar.multiselect('OKUL ADI',u0,default=u0)
    df=df[df['OKUL ADI'].isin(s0)] if s0 else df
    u1=sorted(df['PROGRAM ADI'].dropna().unique().tolist())
    s1=st.sidebar.multiselect('PROGRAM ADI',u1,default=u1)
    df=df[df['PROGRAM ADI'].isin(s1)] if s1 else df
st.title('⚡ DataWizard Dashboard')
st.caption(f'{len(df):,} kayit gosteriliyor')
st.divider()
    mc=st.columns(4)
    mc[0].metric('2023',f"{df['2023'].mean():.2f}")
    mc[1].metric('2024',f"{df['2024'].mean():.2f}")
    mc[2].metric('2025',f"{df['2025'].mean():.2f}")
    mc[3].metric('25/23',f"{df['25/23'].mean():.2f}")
st.divider()
    st.subheader('Bar Grafik')
    bd=df.groupby('OKUL ADI')['2023'].mean().reset_index()
    st.plotly_chart(px.bar(bd,x='OKUL ADI',y='2023',color='2023',color_continuous_scale='Blues'),use_container_width=True)
    st.subheader('Cizgi Grafik')
    st.plotly_chart(px.line(df.reset_index(),x='index',y='2023'),use_container_width=True)
    st.subheader('Scatter Plot')
    st.plotly_chart(px.scatter(df,x='2023',y='2024',color='OKUL ADI',opacity=0.7),use_container_width=True)
with st.expander('Ham Veriyi Goster'):
    st.dataframe(df, use_container_width=True)
    st.download_button('CSV Indir', df.to_csv(index=False).encode('utf-8-sig'), 'veri.csv', 'text/csv')
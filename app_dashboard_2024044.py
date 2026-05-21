#######################
# Import libraries
import streamlit as st
import pandas as pd
import plotly.express as px

#######################
# Page configuration
st.set_page_config(
    page_title="Ireland Agri-Food Dashboard",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded")

#######################
# Load data
df_exports = pd.read_csv('irish_exports_ml.csv')
df_faostat = pd.read_csv('faostat_ml.csv')
df_farm = pd.read_csv('farm_structure_prepared_df.csv')

#######################
# Sidebar
with st.sidebar:
    st.title('🇮🇪 Ireland Agri-Food Dashboard')
    
    selected_view = st.selectbox(
        'Select a view',
        options=[
            'Ireland Agri-Food Exports (2018-2022)',
            'Dairy and Beef Export Trends (2018-2022)',
            'Export Composition for all years',
            'Ireland vs World: Trade Balance',
            'Supervised Learning: Model Accuracy Comparison',
            'Supervised Learning: Cross Validation Results',
            'Unsupervised Learning: KMeans Silhouette Score',
            'Unsupervised Learning: PCA Explained Variance'
        ]
    )
    
#######################
# Plots

# View 1 - Ireland Agri-Food Exports
def make_exports_ireland():
    top_exports = df_exports.groupby('Category')['Amount_EUR'].sum().reset_index()
    top_exports = top_exports.sort_values('Amount_EUR', ascending=False).head(10)
    top_exports = top_exports.sort_values('Amount_EUR', ascending=True)
    top_exports['Amount_EUR_B'] = top_exports['Amount_EUR'] / 1e9
    top_exports['color'] = top_exports['Category'].apply(
        lambda x: '#006400' if x in ['Dairy Produce', 'Beef'] else '#4a90a4'
    )
    fig = px.bar(
        top_exports,
        x='Amount_EUR_B', y='Category', orientation='h',
        title="Ireland's Top 10 Agri-Food Exports by Category (2018-2022)",
        labels={'Amount_EUR_B': 'Total Export Value (€ Billion)', 'Category': ''},
        color='color', color_discrete_map='identity'
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        margin=dict(l=250, r=50, t=50, b=50)
    )
    fig.update_xaxes(showgrid=False, tickprefix='€', ticksuffix='B')
    fig.update_yaxes(showgrid=False)
    return fig
    
# View 2 - Dairy and Beef Export Trends
def make_trends_dairy_beef():
    df_dairy_beef = df_exports[df_exports['Category'].isin(['Dairy Produce', 'Beef'])]
    df_dairy_beef = df_dairy_beef.groupby(['Year', 'Category'])['Amount_EUR'].sum().reset_index()
    df_dairy_beef['Amount_EUR_B'] = df_dairy_beef['Amount_EUR'] / 1e9
    df_dairy_beef['Year'] = df_dairy_beef['Year'].astype(str)
    fig = px.line(
        df_dairy_beef,
        x='Year', y='Amount_EUR_B', color='Category', markers=True,
        title="Ireland's Dairy and Beef Export Trends (2018-2022)",
        labels={'Amount_EUR_B': 'Export Value (€ Billion)', 'Year': 'Year'},
        color_discrete_map={'Dairy Produce': '#006400', 'Beef': '#8B4513'}
    )
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, tickprefix='€', ticksuffix='B')
    return fig
    
# View 3 - Export Composition
def make_export_composition():
    total_by_cat = df_exports.groupby('Category')['Amount_EUR'].sum().sort_values(ascending=False).reset_index()
    top5_pie = total_by_cat.head(5)
    others = pd.DataFrame({'Category': ['Others'], 'Amount_EUR': [total_by_cat['Amount_EUR'][5:].sum()]})
    pie_data = pd.concat([top5_pie, others], ignore_index=True)
    fig = px.pie(
        pie_data,
        names='Category', values='Amount_EUR',
        title="Ireland's Agri-Food Export Composition (2018-2022)",
        color_discrete_sequence=['#006400', '#8B4513', '#4a90a4', '#f4a261', '#e9c46a', '#adb5bd']
    )
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    return fig
    
# View 4 - Ireland vs World Trade Balance
def make_ireland_vs_world():
    trade_by_country = df_faostat.groupby('Country')['trade_balance'].sum().reset_index()
    trade_by_country = trade_by_country.sort_values('trade_balance', ascending=True)
    trade_by_country['color'] = trade_by_country['Country'].apply(
        lambda x: '#006400' if x == 'Ireland' else '#4a90a4'
    )
    fig = px.bar(
        trade_by_country,
        x='trade_balance', y='Country', orientation='h',
        title='Agricultural Trade Balance by Country (Log10 Scaled, 2019-2021)',
        labels={'trade_balance': 'Total Trade Balance (Log10)', 'Country': ''},
        color='color', color_discrete_map='identity'
    )
    fig.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig
    
# View 5 - Supervised Learning Model Accuracy
def make_model_accuracy():
    results_final = []
    results_final.append(['Decision Tree', 66.97, 94.38])
    results_final.append(['Random Forest', 92.17, 94.78])
    results_final.append(['KNN', 91.30, 91.16])
    results_final.append(['GridSearchCV (Decision Tree)', 91.66, 94.38])
    results_df_final = pd.DataFrame(results_final, columns=[
        'Model', 'Scenario 1 - All Products (%)', 'Scenario 2 - Dairy & Beef (%)'
    ])
    fig = px.bar(
        results_df_final,
        x='Model',
        y=['Scenario 1 - All Products (%)', 'Scenario 2 - Dairy & Beef (%)'],
        barmode='group',
        title='Supervised Learning - Model Accuracy Comparison: Scenario 1 vs Scenario 2',
        labels={'value': 'Accuracy (%)', 'variable': 'Scenario'},
        color_discrete_map={
            'Scenario 1 - All Products (%)': '#4a90a4',
            'Scenario 2 - Dairy & Beef (%)': '#006400'
        }
    )
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', yaxis_range=[0, 110])
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig
    
# View 6 - Cross Validation Results
def make_cross_validation():
    cv_comparison = []
    cv_comparison.append(['Decision Tree', 90.92, 88.42])
    cv_comparison.append(['Random Forest', 90.23, 89.99])
    cv_comparison.append(['KNN', 90.95, 90.35])
    cv_comparison_df = pd.DataFrame(cv_comparison, columns=[
        'Model', 'Scenario 1 CV (%)', 'Scenario 2 CV (%)'
    ])
    fig = px.bar(
        cv_comparison_df,
        x='Model',
        y=['Scenario 1 CV (%)', 'Scenario 2 CV (%)'],
        barmode='group',
        title='Supervised Learning - Cross Validation Results: Scenario 1 vs Scenario 2',
        labels={'value': 'CV Accuracy (%)', 'variable': 'Scenario'},
        color_discrete_map={
            'Scenario 1 CV (%)': '#4a90a4',
            'Scenario 2 CV (%)': '#006400'
        }
    )
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', yaxis_range=[0, 110])
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig
    
# View 7 - KMeans Silhouette Score
def make_kmeans():
    results_for_task3 = []
    results_for_task3.append(['Scenario 1: Selecting all the products', 4, 0.4397, 40.07, 27.78])
    results_for_task3.append(['Scenario 2: Selecting only dairy and beef products', 3, 0.4315, 44.48, 26.47])
    results_for_task3_df = pd.DataFrame(results_for_task3, columns=[
        'Scenario', 'Optimal Clusters', 'Silhouette Score', 'PCA Component 1 (%)', 'PCA Component 2 (%)'
    ])
    fig = px.bar(
        results_for_task3_df,
        x='Scenario', y='Silhouette Score',
        title='Unsupervised Learning - KMeans Silhouette Score: Scenario 1 vs Scenario 2',
        labels={'Silhouette Score': 'Silhouette Score', 'Scenario': ''},
        color='Scenario',
        color_discrete_map={
            'Scenario 1: Selecting all the products': '#4a90a4',
            'Scenario 2: Selecting only dairy and beef products': '#006400'
        }
    )
    fig.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white', yaxis_range=[0, 0.6])
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig
    
# View 8 - PCA Explained Variance
def make_pca():
    results_for_task3 = []
    results_for_task3.append(['Scenario 1: Selecting all the products', 4, 0.4397, 40.07, 27.78])
    results_for_task3.append(['Scenario 2: Selecting only dairy and beef products', 3, 0.4315, 44.48, 26.47])
    results_for_task3_df = pd.DataFrame(results_for_task3, columns=[
        'Scenario', 'Optimal Clusters', 'Silhouette Score', 'PCA Component 1 (%)', 'PCA Component 2 (%)'
    ])
    fig = px.bar(
        results_for_task3_df[['Scenario', 'PCA Component 1 (%)', 'PCA Component 2 (%)']],
        x='Scenario',
        y=['PCA Component 1 (%)', 'PCA Component 2 (%)'],
        barmode='group',
        title='Unsupervised Learning - PCA Explained Variance: Scenario 1 vs Scenario 2',
        labels={'value': 'Explained Variance (%)', 'variable': 'Component'},
        color_discrete_map={
            'PCA Component 1 (%)': '#4a90a4',
            'PCA Component 2 (%)': '#006400'
        }
    )
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', yaxis_range=[0, 60])
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig
    
#######################
# Dashboard Main Panel
if selected_view == 'Ireland Agri-Food Exports (2018-2022)':
    st.plotly_chart(make_exports_ireland(), use_container_width=True)

elif selected_view == 'Dairy and Beef Export Trends (2018-2022)':
    st.plotly_chart(make_trends_dairy_beef(), use_container_width=True)

elif selected_view == 'Export Composition for all years':
    st.plotly_chart(make_export_composition(), use_container_width=True)

elif selected_view == 'Ireland vs World: Trade Balance':
    st.plotly_chart(make_ireland_vs_world(), use_container_width=True)

elif selected_view == 'Supervised Learning: Model Accuracy Comparison':
    st.plotly_chart(make_model_accuracy(), use_container_width=True)

elif selected_view == 'Supervised Learning: Cross Validation Results':
    st.plotly_chart(make_cross_validation(), use_container_width=True)

elif selected_view == 'Unsupervised Learning: KMeans Silhouette Score':
    st.plotly_chart(make_kmeans(), use_container_width=True)

elif selected_view == 'Unsupervised Learning: PCA Explained Variance':
    st.plotly_chart(make_pca(), use_container_width=True)
    
    with st.expander('About', expanded=True):
        st.write('''
            - **Project:** CA2 - MSc in Data Analytics - Feb 2026
            - **Author:** Mikel Rodrigo de Paula Pinto
            - **Student ID:** 2024044
            - **Scenario:** Agriculture in the Republic of Ireland
            - **Ireland is the baseline for all comparisons.**
            - **Datasets:** Irish Agri-Food Exports 2018-2022 | Eurostat 2016 | FAOSTAT Trade Data
            - **GitHub:** https://github.com/CCT-Dublin/ca2-sem-1-mikelcctstudent
        ''')
    
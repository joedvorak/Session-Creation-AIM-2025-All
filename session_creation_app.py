import numpy as np
import streamlit as st
import pandas as pd
import cryptpandas as crp
import hmac
import pickle

st.set_page_config(
    page_title="Automated Session Creation Visualization Tool - AIM 2025 (All Communities)",
    page_icon=":material/category_search:",
    layout="wide",
)


st.title("Automated Session Creation Visualization Tool - AIM 2025 (All Communities)")

st.subheader("Data Retrieval Date: January 29, 2025")
st.write("**NOTE:** ***This sort is based on the initial submission of titles, abstracts and session placement before session organization process.***")
st.write("This  includes presentations submitted to all technical communities for AIM 2025.")

# Password Check to unlock abstracts.
def password_entered():
    """Checks whether a password entered by the user is correct."""
    if hmac.compare_digest(st.session_state["password"], st.secrets["access_password"]):
        st.session_state["password_correct"] = True
        del st.session_state["password"]  # Don't store the password.
    else:
        st.session_state["password_correct"] = False

# Show input for password.
st.text_input(
    "Password to view abstracts", type="password", on_change=password_entered, key="password"
)
if "password_correct" in st.session_state:
    # Returns True if the password is validated.
    if st.session_state.get("password_correct", False):
        st.success("Abstracts Unlocked")
    else:
        st.error("😕 Password incorrect")

# Load DataFrames
# Load presentations first.
# Returns True if the password is validated.
if st.session_state.get("password_correct", False):
    df_presentations = crp.read_encrypted(path='encrypted_df.crypt', password=st.secrets['df_password'])
else:
    df_presentations = pd.read_pickle("aim2025_clustered_presentations_no_abstracts.pkl")

# Load the similarity matrix for presentations
df_similarity = pd.read_pickle("oral_app_similarities.pkl")

# Load the analysis results dictionaries
with open('Cluster_analysis_resultsAIM25.pkl', 'rb') as f:
    cluster_analysis_results = pickle.load(f)
    
with open('Original_analysis_resultsAIM25.pkl', 'rb') as f:
    original_analysis_results = pickle.load(f)
# Load the session lists
with open('cluster_session_list.pkl', 'rb') as f:
    cluster_session_list = pickle.load(f)
    
with open('original_session_list.pkl', 'rb') as f:
    original_session_list = pickle.load(f)

# Load the session information
df_clustering_sessions = pd.read_pickle('AI_sessionsAIM25.pkl')
df_sessions = pd.read_pickle('Original_sessionsAIM25.pkl')
df_clustering_session_similarity = pd.read_pickle('Clustering_session_similarityAIM25.pkl')
df_orig_session_similarity = pd.read_pickle('Original_session_similarityAIM25.pkl')


tab_clustered_session, tab_orig_session, tab_pres =  st.tabs(['View Clustered Sessions', 'View Original Sessions', 'View Presentations'])

with tab_clustered_session:
    st.header("Clustered Sessions")
    st.write(f"Average Session Similarity*: {cluster_analysis_results['avg_similarity']:.3f}")
    st.write(f"Minimum Session Similarity*: {cluster_analysis_results['min_session_similarity']:.3f}")
    st.write(f"Number of Sessions: {cluster_analysis_results['num_sessions']}")
    st.write(f"Number of sessions with more than 1 item: {cluster_analysis_results['clusters_gt1']}")
    st.write(f"Presentations placed in sessions: {sum([len(s) for s in cluster_session_list])}")
    st.write(f"*Values do not include sessions with only one item as they cannot have a meaningful similarity score.")
    
    # Plot session size 
    # Get the session sizes dictionary, convert to a DataFrame for better labeling in st.bar_chart
    # The index will be the session names, and the values will be the sizes.
    cluster_session_sizes_dict = cluster_analysis_results['session_sizes']
    cluster_session_sizes_df = pd.DataFrame.from_dict(
        cluster_session_sizes_dict, 
        orient='index', 
        columns=['Presentations Submitted'] # This sets the column name for the bar heights
    )
    cluster_session_sizes_df.index.name = 'Session Name'  # Set the index name for better labeling
    cluster_session_sizes_df.sort_index(inplace=True)  # Sort the index for better visualization
    st.subheader("Session Size Distribution - Clustered Sessions") 
    st.bar_chart(cluster_session_sizes_df, x_label="Session Name", y_label="Presentations Count") 

    # Plot session similarity distribution
    # Get the session similarity dictionary, convert to a DataFrame for better labeling in st.bar_chart
    # The index will be the session names, and the values will be the similarities.
    cluster_session_sim_dict = cluster_analysis_results['similarity_values']
    cluster_session_sim_df = pd.DataFrame.from_dict(
        cluster_session_sim_dict, 
        orient='index', 
        columns=['Session Similiarity'] # This sets the column name for the bar heights
    )
    cluster_session_sim_df.index.name = 'Session Name'  # Set the index name for better labeling
    cluster_session_sim_df.sort_index(inplace=True)  # Sort the index for better visualization
    st.subheader("Session Similarity Distribution - Clustered Sessions") 
    st.bar_chart(cluster_session_sim_df, x_label="Session Name", y_label="Session Similiarity")

    with st.expander('**Instructions** Click to expand'):
        st.write("Select a session by clicking on the checkbox in the leftmost column. Its details and assigned presentations will appear below. You can sort the session list by any column or search for a session name. Just click on the column or mouse over the table.")
    df_clustering_sessions.set_index('Clustering Session', inplace=True)
    event_clustered_session = st.dataframe(
            df_clustering_sessions,
            use_container_width=True,
            column_order=["Clustering Session", 'Gemini Minimal Example Title 1', 'Gemini Minimal Example Keywords', 'Top Committee Match', 'Top Committee Similarity', '2nd Committee Match', '2nd Committee Similarity', '3rd Committee Match', '3rd Committee Similarity', 'Session Similarity - Clustering'],
            column_config={
                "Session Similarity - Clustering" : st.column_config.NumberColumn(format='%.3f'),
                "Top Committee Similarity" : st.column_config.NumberColumn(format='%.3f'),
                "2nd Committee Similarity" : st.column_config.NumberColumn(format='%.3f'),
                "3rd Committee Similarity" : st.column_config.NumberColumn(format='%.3f'),
            },
            on_select="rerun",
            selection_mode="single-row",
        )

    if event_clustered_session.selection.rows: # Check if a session has been selected.
        selected_clustered_session_df = df_clustering_sessions.iloc[event_clustered_session.selection.rows]  # Create a dataframe from the selected session row.
        selected_clustered_session = selected_clustered_session_df.index[0]
        st.header(f"Session {int(selected_clustered_session):d} Session Details")
        selected_clustered_session_df_transposed = selected_clustered_session_df.T
        selected_clustered_session_df_transposed.drop(index=['Session Similarity - Clustering', 'Session Std Dev - Clustering'], inplace=True)

        st.write(f"**Session Similarity:** {selected_clustered_session_df.iloc[0]['Session Similarity - Clustering']:.3f}")
        st.write(f"**Session Similarity Std Dev:** {selected_clustered_session_df.iloc[0]['Session Std Dev - Clustering']:.3f}")
        with st.expander("**Table Details** Click to expand"):
            st.write("This table shows the LLM output for the session. They are named as [LLM][Prompt][Variable].")
            st.write("For example: *Gemini Complete Example Title 1* is the 1st title option provided by the Gemini LLM for the Complete Example prompt.")
        st.dataframe(
            selected_clustered_session_df_transposed,
            use_container_width=True,
        )
        st.write(f"**Presentations in this session**")
        df_selected_clustered_session = df_presentations[df_presentations['Clustering Session'] == selected_clustered_session]
        if 'Abstract' in df_selected_clustered_session: # Check if the dataframe has the abstract in it to determine how to display.
            st.dataframe(
                df_selected_clustered_session,
                use_container_width=True,
                hide_index=True,
                column_order=["Presentation-Session Similarity - Clustering", 'Standardized Deviation - Clustering', 'Abstract ID', 'Title', 'Abstract', 'Original Session'],
                column_config={
                    'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                    "Presentation-Session Similarity - Clustering" : st.column_config.NumberColumn(format='%.3f'),
                    "Session Similarity - Clustering" : None,
                    'Session Std Dev - Clustering': None,
                    "Raw Deviation - Clustering" : st.column_config.NumberColumn(format='%.3f'),
                    "Standardized Deviation - Clustering" : st.column_config.NumberColumn(format='%.3f'),
                },
            )
        else:
            st.dataframe(
                df_selected_clustered_session,
                use_container_width=True,
                hide_index=True,
                column_order=["Presentation-Session Similarity - Clustering", 'Standardized Deviation - Clustering', 'Abstract ID', 'Title', 'Original Session'],
                column_config={
                    'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                    "Presentation-Session Similarity - Clustering" : st.column_config.NumberColumn(format='%.3f'),
                    "Session Similarity - Clustering" : None,
                    'Session Std Dev - Clustering': None,
                    "Raw Deviation - Clustering" : st.column_config.NumberColumn(format='%.3f'),
                    "Standardized Deviation - Clustering" : st.column_config.NumberColumn(format='%.3f'),
                },
            )
        st.header("Most Similar Sessions")
        # Create a Series with the most similar sessions
        similar_clustered_sessions = df_clustering_session_similarity[selected_clustered_session].sort_values(ascending=False) 
        # Remove the selected session itself from the similar sessions
        similar_clustered_sessions = similar_clustered_sessions.drop(selected_clustered_session)
        similar_clustered_sessions_df = pd.DataFrame(similar_clustered_sessions)
        st.write("Other sessions that are most similar to:")
        st.subheader(f"Session {int(similar_clustered_sessions_df.columns[0]):d}")
        st.write("This list is initially sorted by similarity to the selected session.")
        similar_clustered_sessions_df = similar_clustered_sessions_df.rename(columns={
            similar_clustered_sessions_df.columns[0]:'Session-Session Similarity Score',
            })
        similar_clustered_sessions_df.insert(0, "Session Similarity Rank", np.arange(1,similar_clustered_sessions_df.shape[0]+1))
        st.dataframe(
            similar_clustered_sessions_df,
            use_container_width=True,
            hide_index=False,
            )

with tab_orig_session:
    st.header("Original Sessions")
    st.write(f"Average Session Similarity*: {original_analysis_results['avg_similarity']:.3f}")
    st.write(f"Minimum Session Similarity*: {original_analysis_results['min_session_similarity']:.3f}")
    st.write(f"Number of Sessions: {original_analysis_results['num_sessions']}")
    st.write(f"Number of sessions with more than 1 item: {original_analysis_results['clusters_gt1']}")
    st.write(f"Presentations placed in sessions: {sum([len(s) for s in original_session_list])}")
    st.write(f"*Values do not include sessions with only one item as they cannot have a meaningful similarity score.")
    
    # Plot session size 
    # Get the session sizes dictionary, convert to a DataFrame for better labeling in st.bar_chart
    # The index will be the session names, and the values will be the sizes.
    orig_session_sizes_dict = original_analysis_results['session_sizes']
    orig_session_sizes_df = pd.DataFrame.from_dict(
        orig_session_sizes_dict, 
        orient='index', 
        columns=['Presentations Submitted'] # This sets the column name for the bar heights
    )
    orig_session_sizes_df.index.name = 'Session Name'  # Set the index name for better labeling
    orig_session_sizes_df.sort_index(inplace=True)  # Sort the index for better visualization
    st.subheader("Session Size Distribution - Original Sessions") 
    st.bar_chart(orig_session_sizes_df, x_label="Session Name", y_label="Presentations Count") 

    # Plot session similarity distribution
    # Get the session similarity dictionary, convert to a DataFrame for better labeling in st.bar_chart
    # The index will be the session names, and the values will be the similarities.
    orig_session_sim_dict = original_analysis_results['similarity_values']
    orig_session_sim_df = pd.DataFrame.from_dict(
        orig_session_sim_dict, 
        orient='index', 
        columns=['Session Similiarity'] # This sets the column name for the bar heights
    )
    orig_session_sim_df.index.name = 'Session Name'  # Set the index name for better labeling
    orig_session_sim_df.sort_index(inplace=True)  # Sort the index for better visualization
    st.subheader("Session Similarity Distribution - Original Sessions") 
    st.bar_chart(orig_session_sim_df, x_label="Session Name", y_label="Session Similiarity")

    with st.expander('**Instructions** Click to expand'):
        st.write("Select a session by clicking on the checkbox in the leftmost column. Its details and assigned presentations will appear below. You can sort the session list by any column or search for a session name. Just click on the column or mouse over the table.")
    event_session = st.dataframe(
            df_sessions,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Original Session Similarity" : st.column_config.NumberColumn(format='%.3f'),
                "Original Session Std Dev" : st.column_config.NumberColumn(format='%.3f'),
            },
            on_select="rerun",
            selection_mode="single-row",
        )

    if event_session.selection.rows: # Check if a session has been selected.
        st.header('Session Details')
        selected_session_df = df_sessions.iloc[event_session.selection.rows]  # Create a dataframe from the selected session row.
        selected_session =selected_session_df.iloc[0]['Original Session']
        st.subheader(selected_session)
        st.write(f"**Session Similarity:** {selected_session_df.iloc[0]['Original Session Similarity']:.3f}")
        df_selected_session = df_presentations[df_presentations['Original Session'] == selected_session]
        if 'Abstract' in df_selected_session: # Check if the dataframe has the abstract in it to determine how to display.
            st.dataframe(
                df_selected_session,
                use_container_width=True,
                hide_index=True,
                column_order=["Original Presentation-Session Similarity", 'Original Standardized Deviation', 'Abstract ID', 'Title', 'Abstract', ],
                column_config={
                    'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                    "Original Presentation-Session Similarity" : st.column_config.NumberColumn(format='%.3f'),
                    "Original Session Similarity" : None,
                    'Original Session Std Dev': None,
                    "Original Raw Deviation" : st.column_config.NumberColumn(format='%.3f'),
                    "Original Standardized Deviation" : st.column_config.NumberColumn(format='%.3f'),
                },
            )
        else:
            st.dataframe(
                df_selected_session,
                use_container_width=True,
                hide_index=True,
                column_order=["Original Presentation-Session Similarity", 'Original Standardized Deviation', 'Abstract ID', 'Title', ],
                column_config={
                    'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                    "Original Presentation-Session Similarity" : st.column_config.NumberColumn(format='%.3f'),
                    "Original Session Similarity" : None,
                    'Original Session Std Dev': None,
                    "Original Raw Deviation" : st.column_config.NumberColumn(format='%.3f'),
                    "Original Standardized Deviation" : st.column_config.NumberColumn(format='%.3f'),
                },
            )
        st.header("Most Similar Sessions")
        # Create a Series with the  most similar sessions
        similar_sessions = df_orig_session_similarity[selected_session].sort_values(ascending=False) 
        # Remove the selected presentation itself from the similar presentations
        similar_sessions = similar_sessions.drop(selected_session)
        similar_sessions_df = pd.DataFrame(similar_sessions)
        st.write("Other sessions that are most similar to:")
        st.subheader(similar_sessions_df.columns[0])
        st.write("This list is initially sorted by similarity to the selected session.")
        similar_sessions_df = similar_sessions_df.rename(columns={
            similar_sessions_df.columns[0]:'Session-Session Similarity Score',
            })
        similar_sessions_df.insert(0, "Session Similarity Rank", np.arange(1,similar_sessions_df.shape[0]+1))
        st.dataframe(
            similar_sessions_df,
            use_container_width=True,
            hide_index=False,
            )

with tab_pres:
    st.header("Presentations") 
    with st.expander('**Instructions** Click to expand'):
        st.write("Select a presentation by clicking on the checkbox. You can sort the presentation list or search as well.")
        st.write("Once a presentation is selected, its abstract and the ten most similar presentations will appear in a list below.")
        st.write("If you move your mouse over the table, a menu will appear in the top left corner that lets you search within the table or download. Clicking on columns will let you sort by the column too.")
        st.write("If text is cut off, click twice on an cell to see the full text. You can scroll left-right and up-down in the table.")
        st.write("Similarity scores range from 0.0 (not similar) to 1.0 (identical).")
        st.write("The leftmost column is a checkbox column. Click to select a presentation. This may blend with the background on dark themes.")

    event = st.dataframe(
            df_presentations,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                "Original Presentation-Session Similarity" : st.column_config.NumberColumn(format='%.3f'),
                "Original Session Similarity" : None,
                'Original Session Std Dev': None,
                "Original Raw Deviation" : st.column_config.NumberColumn(format='%.3f'),
                "Original Standardized Deviation" : st.column_config.NumberColumn(format='%.3f'),
            },
            on_select="rerun",
            selection_mode="single-row",
        )


    if event.selection.rows: # Check if a presentation has been selected.
        st.header("Selected Presentation:")
        selected_pres = df_presentations.iloc[event.selection.rows]  # Create a dataframe from the selected presentation row.
        st.write(selected_pres.iloc[0]['Title'])  # It is necessary to request the first row, [0], since it is a dataframe and not just one entry.
        st.header("Most Similar Presentations")
        similar_presentations = df_similarity.loc[selected_pres.iloc[0].name].sort_values(ascending=False) # Create a Series with the  most similar presentations
        # Remove the selected presentation itself from the similar presentations
        similar_presentations = similar_presentations.drop(selected_pres.iloc[0].name)
        # Build the similarity dataframe. Add the similarity score and similarity rank to the dataframe and show it.
        similar_df = df_presentations.loc[similar_presentations.index]
        similar_df.insert(0, "Similarity Score", similar_presentations)
        similar_df.insert(0, "Similarity Rank", np.arange(1,similar_df.shape[0]+1))
        st.dataframe(
            similar_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                "Presentation-Session Similarity" : None,
                "Session Similarity" : None,
                'Session Std Dev': None,
                "Raw Deviation" : None,
                "Standardized Deviation" : None,
            },
            )
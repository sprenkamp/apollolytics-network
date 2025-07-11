import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
import warnings
import networkx as nx
from pyvis.network import Network
import matplotlib.colors as mcolors
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# Create database connection
engine = create_engine(f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}')

def analyze_known_forwards():
    """
    Analyze messages that were forwarded from known chats
    """
    # Get all unique chat_ids and names from the database (our known chats)
    query_known_chats = """
    SELECT DISTINCT chat_id, chat_name
    FROM relevant_classified_narrative_results 
    WHERE chat_id IS NOT NULL
    """
    
    known_chats_df = pd.read_sql(query_known_chats, engine)
    known_ids = set(known_chats_df['chat_id'].tolist())
    known_chat_names = dict(zip(known_chats_df['chat_id'], known_chats_df['chat_name']))
    
    print(f"Found {len(known_ids)} known chat IDs")
    
    # Get all messages with forward information
    query_forwards = """
    SELECT id, chat_id, chat_name, messagedatetime, messagetext, fwd_from, fwd_from_type,
           denazificationofukraine_similarity, protectionofrussianspeakers_similarity,
           natoexpansionthreat_similarity, biolabsconspiracy_similarity, historicalunity_similarity,
           westernrussophobia_similarity, sanctionsaseconomicwarfare_similarity,
           legitimizingannexedterritories_similarity, discreditingukraineleadership_similarity,
           putinsdeath_similarity, russiascollapse_similarity, nordstreampipelinesabotage_similarity,
           heroicmyths_similarity, optimismstrategy_similarity, notsidingwithukraine_similarity,
           ukraineagainstnewfascism_similarity, cannonfodder_similarity, truthvsrussianlies_similarity
    FROM relevant_classified_narrative_results 
    WHERE fwd_from IS NOT NULL 
    """
    
    forwards_df = pd.read_sql(query_forwards, engine)
    
    print(f"Found {len(forwards_df)} messages with forward information")
    
    # Filter for high similarity scores (> 0.875) in any narrative column
    similarity_columns = [
        'denazificationofukraine_similarity', 'protectionofrussianspeakers_similarity',
        'natoexpansionthreat_similarity', 'biolabsconspiracy_similarity', 'historicalunity_similarity',
        'westernrussophobia_similarity', 'sanctionsaseconomicwarfare_similarity',
        'legitimizingannexedterritories_similarity', 'discreditingukraineleadership_similarity',
        'putinsdeath_similarity', 'russiascollapse_similarity', 'nordstreampipelinesabotage_similarity',
        'heroicmyths_similarity', 'optimismstrategy_similarity', 'notsidingwithukraine_similarity',
        'ukraineagainstnewfascism_similarity', 'cannonfodder_similarity', 'truthvsrussianlies_similarity'
    ]
    
    # Create mask for high similarity scores
    high_similarity_mask = forwards_df[similarity_columns].max(axis=1) > 0.875
    forwards_df = forwards_df[high_similarity_mask]
    
    print(f"After filtering for similarity > 0.875: {len(forwards_df)} messages")
    
    # Filter messages that involve known chats (either as source or target)
    known_forwards = []
    
    for _, row in forwards_df.iterrows():
        try:
            # Extract the original chat ID from fwd_from
            orig_chat_id = int(row['fwd_from'])
            current_chat_id = row['chat_id']
            
            # Include if either source OR target is known
            source_is_known = orig_chat_id in known_ids
            target_is_known = current_chat_id in known_ids
            
            if source_is_known or target_is_known:
                # Get chat names
                if source_is_known:
                    orig_chat_name = known_chat_names.get(orig_chat_id, "Unknown")
                else:
                    orig_chat_name = f"Unknown_Source_{orig_chat_id}"
                
                if target_is_known:
                    current_chat_name = row['chat_name']
                else:
                    current_chat_name = f"Unknown_Target_{current_chat_id}"
                
                known_forwards.append({
                    'message_id': row['id'],
                    'current_chat_id': current_chat_id,
                    'current_chat_name': current_chat_name,
                    'original_chat_id': orig_chat_id,
                    'original_chat_name': orig_chat_name,
                    'datetime': row['messagedatetime'],
                    'text': row['messagetext'][:100] + '...' if len(str(row['messagetext'])) > 100 else row['messagetext']
                })
                
                if source_is_known and target_is_known:
                    print(f"Message {row['id']} was forwarded from '{orig_chat_name}' (id={orig_chat_id}) to '{current_chat_name}'")
                elif source_is_known:
                    print(f"Message {row['id']} was forwarded from '{orig_chat_name}' (id={orig_chat_id}) to unknown chat (id={current_chat_id})")
                else:
                    print(f"Message {row['id']} was forwarded from unknown chat (id={orig_chat_id}) to '{current_chat_name}'")
                    
        except (ValueError, TypeError):
            # Skip if fwd_from is not a valid integer
            continue
    
    print(f"\nTotal messages involving known chats: {len(known_forwards)}")
    
    if known_forwards:
        # Convert to DataFrame for easier analysis
        known_forwards_df = pd.DataFrame(known_forwards)
        
        # Basic statistics
        print("\nForward statistics:")
        print(f"Unique source chats: {known_forwards_df['original_chat_id'].nunique()}")
        print(f"Unique destination chats: {known_forwards_df['current_chat_id'].nunique()}")
        
        return known_forwards_df
    
    return known_forwards_df

def create_known_forwards_network(known_forwards_df):
    """
    Create network visualization for known forwards
    """
    if known_forwards_df is None or known_forwards_df.empty:
        print("No known forwards data to visualize")
        return
    
    print("Creating network visualization for known forwards...")
    
    # Create network graph
    G = nx.MultiDiGraph()
    
    # Define colors for different chat types
    node_colors = {
        'channel': '#FF6B6B',      # Red for channels
        'group': '#4ECDC4',        # Teal for groups
        'unknown': '#FFA500'       # Orange for unknown
    }
    
    # Create mapping from chat_id to country for known chats
    chat_id_to_country = {}
    for _, row in known_forwards_df.iterrows():
        # Get country from the original query data
        # We need to query the database to get country information
        pass
    
    # Add nodes for all unique chats (including unknown ones)
    all_chats = set()
    unknown_source_chats = set()
    unknown_target_chats = set()
    
    for _, row in known_forwards_df.iterrows():
        all_chats.add(row['current_chat_name'])
        all_chats.add(row['original_chat_name'])
        
        # Track unknown chats
        if row['original_chat_name'].startswith('Unknown_Source_'):
            unknown_source_chats.add(row['original_chat_name'])
        if row['current_chat_name'].startswith('Unknown_Target_'):
            unknown_target_chats.add(row['current_chat_name'])
    
    # Query database to get country information for all chat_ids
    all_chat_ids = set()
    for _, row in known_forwards_df.iterrows():
        all_chat_ids.add(row['current_chat_id'])
        all_chat_ids.add(row['original_chat_id'])
    
    # Get country information from database
    chat_ids_str = ','.join([str(cid) for cid in all_chat_ids])
    country_query = f"""
    SELECT DISTINCT chat_id, country 
    FROM relevant_classified_narrative_results 
    WHERE chat_id IN ({chat_ids_str})
    """
    country_df = pd.read_sql(country_query, engine)
    chat_id_to_country = dict(zip(country_df['chat_id'], country_df['country']))
    
    # Add nodes to graph
    for chat in all_chats:
        # Handle unknown chats
        if chat.startswith('Unknown_Source_') or chat.startswith('Unknown_Target_'):
            if chat.startswith('Unknown_Source_'):
                node_type = "Unknown Source"
                node_color = '#FFA500'  # Orange for unknown sources
                # Extract the actual chat ID
                chat_id = chat.replace('Unknown_Source_', '')
            else:
                node_type = "Unknown Target"
                node_color = '#9370DB'  # Purple for unknown targets
                # Extract the actual chat ID
                chat_id = chat.replace('Unknown_Target_', '')
            
            G.add_node(chat, label=f"ID: {chat_id}", title=f"Chat ID: {chat_id} ({node_type})", 
                      size=15, color=node_color, node_type=node_type, country="Unknown")
            continue
        
        # Handle known chats
        # Find the chat_id for this chat name
        chat_id = None
        for _, row in known_forwards_df.iterrows():
            if row['current_chat_name'] == chat:
                chat_id = row['current_chat_id']
                break
            elif row['original_chat_name'] == chat:
                chat_id = row['original_chat_id']
                break
        
        # Determine node type and country
        if 'channel' in chat.lower() or 'news' in chat.lower():
            node_type = "Channel"
        else:
            node_type = "Group"
        
        # Determine country and color
        country = chat_id_to_country.get(chat_id, "Unknown")
        if country == "Russia":
            if node_type == "Channel":
                node_color = '#FF4444'  # Dark red for Russian channels
            else:
                node_color = '#FF8888'  # Light red for Russian groups
        elif country == "Ukraine":
            if node_type == "Channel":
                node_color = '#4444FF'  # Dark blue for Ukrainian channels
            else:
                node_color = '#8888FF'  # Light blue for Ukrainian groups
        else:
            if node_type == "Channel":
                node_color = node_colors['channel']  # Default channel color
            else:
                node_color = node_colors['group']    # Default group color
        
        G.add_node(chat, label=chat, title=f"{chat} ({node_type}, {country})", 
                  size=20, color=node_color, node_type=node_type, country=country)
    
    # Add edges for each forward
    for _, row in known_forwards_df.iterrows():
        source = row['original_chat_name']
        target = row['current_chat_name']
        
        G.add_edge(source, target, 
                  message_id=row['message_id'],
                  datetime=row['datetime'],
                  text=row['text'])
    
    # Create PyVis network
    net = Network(height='750px', width='100%', notebook=False, directed=True)
    net.force_atlas_2based()
    
    # Add nodes
    for node, data in G.nodes(data=True):
        net.add_node(node, 
                    label=data.get('label', node), 
                    title=data.get('title', node), 
                    size=data.get('size', 15),
                    color=data.get('color', '#C0C0C0'))
    
    # Add edges
    for source, target, data in G.edges(data=True):
        message_id = data.get('message_id', 'unknown')
        datetime = data.get('datetime', 'unknown')
        text = data.get('text', '')
        
        tooltip = f"<b>Message ID:</b> {message_id}<br><b>Date:</b> {datetime}<br><b>Text:</b> {text}"
        
        net.add_edge(source, target,
                    title=tooltip,
                    color='#666666',
                    width=1)
    
    # Add legend
    legend_html = """
    <div style="position: absolute; top: 10px; left: 10px; background: white; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
        <h4>Known Forwards Network</h4>
        <p><b>Nodes:</b> Chats/Channels</p>
        <p><b>Edges:</b> Forwarded messages between known chats</p>
        <p><b>Direction:</b> Arrow shows forwarding direction</p>
        <h5>Node Colors by Country & Type:</h5>
        <div style="color: #FF4444; font-weight: bold;">● Russian Channels</div>
        <div style="color: #FF8888; font-weight: bold;">● Russian Groups</div>
        <div style="color: #4444FF; font-weight: bold;">● Ukrainian Channels</div>
        <div style="color: #8888FF; font-weight: bold;">● Ukrainian Groups</div>
        <div style="color: #FFA500; font-weight: bold;">● Unknown Source Chats</div>
        <div style="color: #9370DB; font-weight: bold;">● Unknown Target Chats</div>
        <h5>Statistics:</h5>
        <div>Total forwards: {}</div>
        <div>Unique source chats: {}</div>
        <div>Unique target chats: {}</div>
        <div>Unknown source chats: {}</div>
        <div>Unknown target chats: {}</div>
    </div>
    """.format(len(known_forwards_df), 
               known_forwards_df['original_chat_id'].nunique(),
               known_forwards_df['current_chat_id'].nunique(),
               len(unknown_source_chats),
               len(unknown_target_chats))
    
    # Save network
    net.save_graph('temp_network.html')
    
    # Add legend to HTML
    with open('temp_network.html', 'r') as f:
        html = f.read()
    
    html = html.replace('</body>', legend_html + '</body>')
    
    output_file = 'known_forwards_network.html'
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Network visualization saved to {output_file}")
    
    # Print network statistics
    print(f"\nNetwork Statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Total forwards: {len(known_forwards_df)}")
    
    return G

if __name__ == "__main__":
    known_forwards_df = analyze_known_forwards()
    if known_forwards_df is not None:
        create_known_forwards_network(known_forwards_df)


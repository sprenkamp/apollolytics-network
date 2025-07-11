import pandas as pd
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv
import numpy as np

"""
Script: 02_networks.py

This script builds interactive network graphs of narrative propagation for messages above 0.875 similarity threshold.
It analyzes how narratives spread between channels and groups, with time-based highlighting and narrative-specific coloring.

- Input: relevant_classified_narrative_results table with narrative similarity scores
- Output: Interactive HTML network visualizations showing narrative propagation patterns

Usage:
    python 02_networks.py
"""

# Load environment variables
load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# Configuration
SIMILARITY_THRESHOLD = 0.875
AGGREGATION_PERIOD = 'monthly'  # Options: 'daily', 'weekly', 'monthly'

# Connect to database
engine = create_engine(
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

def get_narrative_columns(df):
    """Get all narrative-related columns from the dataframe"""
    return [col for col in df.columns if col.lower().endswith('_similarity')]

def get_country_narratives():
    """Define which narratives belong to which country"""
    russian_narratives = [
        'denazificationofukraine_similarity',
        'protectionofrussianspeakers_similarity', 
        'natoexpansionthreat_similarity',
        'biolabsconspiracy_similarity',
        'historicalunity_similarity',
        'westernrussophobia_similarity',
        'sanctionsaseconomicwarfare_similarity',
        'legitimizingannexedterritories_similarity',
        'discreditingukraineleadership_similarity'
    ]
    
    ukrainian_narratives = [
        'putinsdeath_similarity',
        'russiascollapse_similarity',
        'nordstreampipelinesabotage_similarity',
        'heroicmyths_similarity',
        'optimismstrategy_similarity',
        'notsidingwithukraine_similarity',
        'ukraineagainstnewfascism_similarity',
        'cannonfodder_similarity',
        'truthvsrussianlies_similarity'
    ]
    
    return russian_narratives, ukrainian_narratives

def load_narrative_data():
    """Load the classified narrative results from the database"""
    print("Loading narrative data from database...")
    df = pd.read_sql("SELECT * FROM relevant_classified_narrative_results", engine)
    print(f"Loaded {len(df)} rows of data")
    
    # Filter for relevant narratives (similarity > threshold)
    similarity_cols = get_narrative_columns(df)
    if similarity_cols:
        # Keep only rows where any narrative similarity > threshold
        mask = (df[similarity_cols] > SIMILARITY_THRESHOLD).any(axis=1)
        df = df[mask]
        print(f"Filtered to {len(df)} rows with narrative similarity > {SIMILARITY_THRESHOLD}")
    
    return df

def create_message_sharing_network(df, country_filter=None, output_file=None):
    """Create network visualization for actual message sharing/forwarding"""
    if country_filter:
        print(f"Creating message sharing network for {country_filter}...")
        # Filter data for the specific country
        filtered_df = df[df['country'] == country_filter].copy()
        if filtered_df.empty:
            print(f"No data found for {country_filter}")
            return None
    else:
        print("Creating combined message sharing network for all countries...")
        filtered_df = df.copy()
    
    # Get all narratives
    russian_narratives, ukrainian_narratives = get_country_narratives()
    all_narratives = russian_narratives + ukrainian_narratives
    
    # Filter to only include available narratives
    available_narratives = [col for col in all_narratives if col in filtered_df.columns]
    
    if not available_narratives:
        print(f"No narrative data available")
        return None
    
    # Check if we have forwarding data
    if 'fwd_from' not in filtered_df.columns:
        print("No forwarding data available. Cannot create message sharing network.")
        return None
    
    # Filter for forwarded messages only
    forwarded_df = filtered_df[filtered_df['fwd_from'].notna()].copy()
    
    if forwarded_df.empty:
        print(f"No forwarded messages found")
        return None
    
    print(f"Found {len(forwarded_df)} forwarded messages")
    
    # Convert date column to datetime
    date_col = None
    for col in ['messagedatetime', 'messagedate', 'date', 'datetime', 'timestamp', 'created_at']:
        if col in filtered_df.columns:
            date_col = col
            break
    
    if date_col is None:
        print(f"No date column found")
        return None
    
    forwarded_df[date_col] = pd.to_datetime(forwarded_df[date_col], errors='coerce')
    forwarded_df = forwarded_df.dropna(subset=[date_col])
    
    # Create aggregation period column
    if AGGREGATION_PERIOD == 'daily':
        forwarded_df['period'] = forwarded_df[date_col].dt.date
    elif AGGREGATION_PERIOD == 'weekly':
        forwarded_df['period'] = forwarded_df[date_col].dt.to_period('W-MON').dt.start_time
    elif AGGREGATION_PERIOD == 'monthly':
        forwarded_df['period'] = forwarded_df[date_col].dt.to_period('M').dt.start_time
    
    # Create network graph (directed for forwarding) - use MultiDiGraph for individual edges
    G = nx.MultiDiGraph()
    
    # Define colors for narratives
    cmap = plt.get_cmap('tab20')
    narrative_colors = {}
    for i, narrative in enumerate(available_narratives):
        narrative_name = narrative.replace('_similarity', '').replace('_', ' ')
        narrative_colors[narrative_name] = mcolors.rgb2hex(cmap(i % cmap.N))
    
    # Create mapping from chat_id to chat_name and type (using the same approach as 05_known_forwards.py)
    chat_id_to_name = {}
    chat_id_to_type = {}
    for _, row in filtered_df.iterrows():
        if pd.notna(row.get('chat_id')) and pd.notna(row.get('chat_name')):
            chat_id_to_name[str(row['chat_id'])] = row['chat_name']
            if pd.notna(row.get('type')):
                chat_id_to_type[str(row['chat_id'])] = row['type'].lower()
    
    # Add nodes for each chat/channel
    all_chats = set()
    unknown_source_chats = set()
    unknown_target_chats = set()
    
    for _, row in forwarded_df.iterrows():
        source_chat_id = str(row['fwd_from'])
        target_chat_id = str(row['chat_id'])
        
        source_chat = chat_id_to_name.get(source_chat_id, f"Unknown_Source_{source_chat_id}")
        target_chat = chat_id_to_name.get(target_chat_id, f"Unknown_Target_{target_chat_id}")
        
        if source_chat.startswith("Unknown_Source_"):
            unknown_source_chats.add(source_chat_id)
        if target_chat.startswith("Unknown_Target_"):
            unknown_target_chats.add(target_chat_id)
        
        all_chats.add(source_chat)
        all_chats.add(target_chat)
    
    # Print statistics about unknown chats
    if unknown_source_chats:
        print(f"  Warning: {len(unknown_source_chats)} unknown source chats (messages forwarded FROM these chats)")
        print(f"    Example chat_ids: {list(unknown_source_chats)[:5]}")
    if unknown_target_chats:
        print(f"  Warning: {len(unknown_target_chats)} unknown target chats (messages forwarded TO these chats)")
        print(f"    Example chat_ids: {list(unknown_target_chats)[:5]}")
    
    # Define node colors based on chat type
    node_colors = {
        'channel': '#FF6B6B',      # Red for channels (news outlets)
        'group': '#4ECDC4',        # Teal for groups (user communities)
        'unknown_source': '#FFA500', # Orange for unknown sources
        'unknown_target': '#9370DB'  # Purple for unknown targets
    }
    
    for chat in all_chats:
        # Determine node color based on chat type
        if chat.startswith("Unknown_Source_"):
            node_color = node_colors['unknown_source']
            node_type = "Unknown Source"
        elif chat.startswith("Unknown_Target_"):
            node_color = node_colors['unknown_target']
            node_type = "Unknown Target"
        else:
            # Find the chat_id for this chat name
            chat_id = None
            for cid, chat_name in chat_id_to_name.items():
                if chat_name == chat:
                    chat_id = cid
                    break
            
            if chat_id and chat_id in chat_id_to_type:
                chat_type = chat_id_to_type[chat_id]
                if chat_type == 'channel':
                    node_color = node_colors['channel']
                    node_type = "Channel"
                elif chat_type == 'group':
                    node_color = node_colors['group']
                    node_type = "Group"
                else:
                    node_color = node_colors['group']  # Default to group
                    node_type = "Group"
            else:
                node_color = node_colors['group']  # Default to group
                node_type = "Group"
        
        G.add_node(chat, label=chat, title=f"{chat} ({node_type})", size=20, color=node_color, node_type=node_type)
    
    # Add edges based on actual message forwarding
    edges_data = []
    
    # Debug counters
    total_forwarded = len(forwarded_df)
    with_narrative_above_threshold = 0
    without_narrative_above_threshold = 0
    
    for _, row in forwarded_df.iterrows():
        source_chat_id = str(row['fwd_from'])
        target_chat_id = str(row['chat_id'])
        
        source_chat = chat_id_to_name.get(source_chat_id, f"Unknown_Source_{source_chat_id}")
        target_chat = chat_id_to_name.get(target_chat_id, f"Unknown_Target_{target_chat_id}")
        
        # Find the dominant narrative for this forwarded message
        max_similarity = 0
        dominant_narrative = None
        
        for narrative_col in available_narratives:
            if narrative_col in row and pd.notna(row[narrative_col]):
                if row[narrative_col] > max_similarity:
                    max_similarity = row[narrative_col]
                    dominant_narrative = narrative_col.replace('_similarity', '').replace('_', ' ')
        

        
        # Always create edge for forwarded message, regardless of narrative
        message_id = row.get('id', 'unknown')
        message_text = row.get('messagetext', '')[:100] + '...' if len(str(row.get('messagetext', ''))) > 100 else str(row.get('messagetext', ''))
        period = str(row['period'])
        
        # Determine source and target types using the type column
        if source_chat.startswith("Unknown_Source_"):
            source_type = 'unknown'
        else:
            if source_chat_id in chat_id_to_type:
                source_type = chat_id_to_type[source_chat_id]
            else:
                source_type = 'group'  # Default to group
            
        if target_chat.startswith("Unknown_Target_"):
            target_type = 'unknown'
        else:
            if target_chat_id in chat_id_to_type:
                target_type = chat_id_to_type[target_chat_id]
            else:
                target_type = 'group'  # Default to group
        
        # Use dominant narrative for coloring, or default to gray if none found
        if dominant_narrative and max_similarity > SIMILARITY_THRESHOLD:
            edge_color = narrative_colors.get(dominant_narrative, '#C0C0C0')
            narrative_label = dominant_narrative
            with_narrative_above_threshold += 1
        else:
            edge_color = '#C0C0C0'  # Gray for no dominant narrative
            narrative_label = 'No dominant narrative'
            without_narrative_above_threshold += 1
        
        edges_data.append({
            'source': source_chat,
            'target': target_chat,
            'narrative': narrative_label,
            'period': period,
            'message_id': message_id,
            'message_text': message_text,
            'color': edge_color,
            'source_type': source_type,
            'target_type': target_type
        })
    
    # Debug output
    print(f"  Total forwarded messages: {total_forwarded}")
    print(f"  Messages with narrative above {SIMILARITY_THRESHOLD}: {with_narrative_above_threshold}")
    print(f"  Messages without narrative above {SIMILARITY_THRESHOLD}: {without_narrative_above_threshold}")
    print(f"  Creating {len(edges_data)} edges")
    
    # Add individual edges for each forwarded message
    for edge_data in edges_data:
        source = edge_data['source']
        target = edge_data['target']
        
        if G.has_node(source) and G.has_node(target):
            # Add individual edge for each forwarded message
            G.add_edge(source, target, 
                      message_id=edge_data['message_id'],
                      narrative=edge_data['narrative'],
                      period=edge_data['period'],
                      message_text=edge_data['message_text'],
                      color=edge_data['color'],
                      width=1,
                      source_type=edge_data['source_type'],
                      target_type=edge_data['target_type'])
    
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
        narrative = data.get('narrative', 'unknown')
        period = data.get('period', 'unknown')
        message_text = data.get('message_text', '')
        
        # Create tooltip for individual message
        source_type = data.get('source_type', 'unknown')
        target_type = data.get('target_type', 'unknown')
        tooltip = f"<b>Message ID:</b> {message_id}<br><b>Narrative:</b> {narrative}<br><b>Period:</b> {period}<br><b>Flow:</b> {source_type.title()} → {target_type.title()}<br><b>Text:</b> {message_text}"
        
        edge_color = data.get('color', '#C0C0C0')
        
        net.add_edge(source, target,
                    title=tooltip,
                    color=edge_color,
                    width=1)  # Each edge represents one message
    
    # Add legend
    network_title = f"{country_filter} Message Sharing Network" if country_filter else "Combined Message Sharing Network"
    legend_html = f"""
    <div style="position: absolute; top: 10px; left: 10px; background: white; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
        <h4>{network_title}</h4>
        <p><b>Node size:</b> Chat/Channel importance</p>
        <p><b>Edge width:</b> Each edge represents one forwarded message</p>
        <p><b>Edge color:</b> Dominant narrative in forwarded messages</p>
        <p><b>Direction:</b> Arrow shows forwarding direction</p>
        <h5>Node Types:</h5>
        <div style="color: #FF6B6B; font-weight: bold;">● Channels (News Outlets)</div>
        <div style="color: #4ECDC4; font-weight: bold;">● Groups (User Communities)</div>
        <div style="color: #FFA500; font-weight: bold;">● Unknown Sources</div>
        <div style="color: #9370DB; font-weight: bold;">● Unknown Targets</div>
        <h5>Propagation Analysis:</h5>
        <div style="font-size: 12px;">
        <div>🔴 Channel → Group: Propaganda flow</div>
        <div>🟢 Group → Channel: Feedback</div>
        <div>🟡 Channel → Channel: Coordination</div>
        <div>🔵 Group → Group: Amplification</div>
        </div>
        <h5>Narratives:</h5>
    """
    
    for narrative, color in narrative_colors.items():
        legend_html += f'<div style="color: {color}; font-weight: bold;">■ {narrative}</div>'
    
    legend_html += "</div>"
    
    # Save network
    net.save_graph('temp_network.html')
    
    # Add legend to HTML
    with open('temp_network.html', 'r') as f:
        html = f.read()
    
    html = html.replace('</body>', legend_html + '</body>')
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(html)
        print(f"Message sharing network saved to {output_file}")
    else:
        print("Network created but no output file specified")
    
    # Analyze propagation patterns
    channel_to_group = 0
    group_to_channel = 0
    channel_to_channel = 0
    group_to_group = 0
    
    for source, target, data in G.edges(data=True):
        source_type = data.get('source_type', 'unknown')
        target_type = data.get('target_type', 'unknown')
        
        if source_type == 'channel' and target_type == 'group':
            channel_to_group += 1
        elif source_type == 'group' and target_type == 'channel':
            group_to_channel += 1
        elif source_type == 'channel' and target_type == 'channel':
            channel_to_channel += 1
        elif source_type == 'group' and target_type == 'group':
            group_to_group += 1
    
    # Print network statistics
    network_name = country_filter if country_filter else "Combined"
    print(f"Message sharing network statistics for {network_name}:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Total messages forwarded: {G.number_of_edges()}")
    print(f"  Propagation patterns:")
    print(f"    Channel → Group: {channel_to_group} messages (propaganda flow)")
    print(f"    Group → Channel: {group_to_channel} messages (feedback)")
    print(f"    Channel → Channel: {channel_to_channel} messages (coordination)")
    print(f"    Group → Group: {group_to_group} messages (amplification)")
    
    if channel_to_group > 0:
        propaganda_ratio = channel_to_group / (channel_to_group + group_to_channel + channel_to_channel + group_to_group) * 100
        print(f"    Propaganda flow (Channel→Group): {propaganda_ratio:.1f}% of all forwards")
    
    return G

def main():
    """Main function to create narrative networks"""
    print("Creating narrative propagation networks...")
    
    # Load data
    df = load_narrative_data()
    
    if df.empty:
        print("No data loaded. Exiting.")
        return
    
    # Create output directory
    os.makedirs('network_graphs', exist_ok=True)
    
    # Create message sharing networks for each country and combined
    countries = ['Russia', 'Ukraine']
    
    for country in countries:
        output_file = f'network_graphs/message_sharing_network_{country.lower()}.html'
        create_message_sharing_network(df, country, output_file)
    
    # Create combined network (all countries)
    output_file = 'network_graphs/message_sharing_network_combined.html'
    create_message_sharing_network(df, None, output_file)
    
    print("Network analysis complete! Check the network_graphs directory for HTML files.")

if __name__ == "__main__":
    main()

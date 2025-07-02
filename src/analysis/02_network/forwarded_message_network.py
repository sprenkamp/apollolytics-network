import pandas as pd
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import argparse
import psycopg2
from dotenv import load_dotenv

"""
Script: forwarded_message_network.py

This script builds interactive network graphs of forwarded messages for Russian, Ukrainian, and combined Telegram channels.
It uses PyVis for visualization, with a month slider to highlight edges by time period and color by topic.

- Input: PostgreSQL tables with columns: peer_id, fwd_from, id, messagetext, messagedate, forwards, chat, topic, country
- Output: Three HTML files: network_ru.html, network_ua.html, network_all.html

Usage:
    python forwarded_message_network.py

You can adapt the filtering logic if you do not have a 'country' column.
"""

# Load environment variables from .env file
load_dotenv()
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")


def load_table_as_df(table_name):
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
    conn.close()
    return df

def visualize_forwarded_messages_with_highlight(df, output_file='interactive_graph.html', time_unit='month'):
    df = df.copy()
    # Ensure messagedate is in datetime format
    df['messagedate'] = pd.to_datetime(df['messagedate'], errors='coerce')
    # Ensure 'topic' column is present
    if 'topic' not in df.columns:
        df['topic'] = 'unknown'
    # Step 1: Filter to include only forwarded messages
    forwarded_messages = df[df['fwd_from'].notna()]
    # Define time periods with year included
    if time_unit == 'month':
        forwarded_messages['time_period'] = forwarded_messages['messagedate'].dt.to_period('M')
    elif time_unit == 'week':
        forwarded_messages['time_period'] = forwarded_messages['messagedate'].dt.to_period('W')
    else:
        raise ValueError("time_unit must be 'month' or 'week'.")
    time_periods = sorted(forwarded_messages['time_period'].dropna().unique())
    time_periods_str = [str(tp) for tp in time_periods]
    # Step 2: Assign unique colors to each unique topic
    unique_topics = forwarded_messages['topic'].unique()
    cmap = plt.get_cmap('tab20')
    num_colors = len(unique_topics)
    if num_colors > cmap.N:
        cmap = plt.get_cmap('hsv')
    colors = [mcolors.rgb2hex(cmap(i / num_colors)) for i in range(num_colors)]
    topic_to_color = {topic: colors[i] for i, topic in enumerate(unique_topics)}
    default_edge_color = '#C0C0C0'
    # Step 3: Initialize a directed MultiDiGraph
    G = nx.MultiDiGraph()
    # Step 4: Add edges for forwarded messages
    peer_id_to_chat = df[['peer_id', 'chat']].drop_duplicates().set_index('peer_id')['chat'].to_dict() if 'chat' in df.columns else {}
    for _, row in forwarded_messages.iterrows():
        source_peer_id = row['fwd_from']
        target_peer_id = row['peer_id']
        message_id = row['id']
        message_text = row.get('messagetext', '') if pd.notna(row.get('messagetext', '')) else ''
        message_date = row['messagedate'].strftime('%Y-%m-%d') if pd.notna(row['messagedate']) else ''
        time_period = str(row['time_period'])
        topic = row['topic']
        source_chat = peer_id_to_chat.get(source_peer_id, "unknown name")
        target_chat = peer_id_to_chat.get(target_peer_id, "unknown name")
        source_node = str(source_peer_id)
        target_node = str(target_peer_id)
        if not G.has_node(source_node):
            G.add_node(source_node, label=source_chat, title=source_chat)
        if not G.has_node(target_node):
            G.add_node(target_node, label=target_chat, title=target_chat)
        G.add_edge(
            source_node,
            target_node,
            message_id=message_id,
            text=message_text,
            date=message_date,
            time_period=time_period,
            topic=topic,
            color=default_edge_color,
            width=1
        )
    # Step 5: Create a PyVis Network
    net = Network(height='750px', width='100%', notebook=False, directed=True)
    net.force_atlas_2based()
    for node, data in G.nodes(data=True):
        label = data.get('label', 'unknown name')
        title = data.get('title', '')
        net.add_node(node, label=label, title=title, size=15)
    for source, target, data in G.edges(data=True):
        message_id = data.get('message_id', '')
        text = data.get('text', '')
        date = data.get('date', '')
        time_period = data.get('time_period', '')
        topic = data.get('topic', None)
        tooltip = f"<b>Message ID:</b> {message_id}<br><b>Date:</b> {date}<br><b>Topic:</b> {topic}<br><b>Text:</b> {text}"
        net.add_edge(
            source,
            target,
            title=tooltip,
            time_period=time_period,
            topic=topic,
            color=data.get('color', default_edge_color),
            width=data.get('width', 1)
        )
    net.show_buttons(filter_=['physics'])
    topic_to_color_js = "{\n" + ",\n".join([f"'{str(k)}': '{v}'" for k, v in topic_to_color.items()]) + "\n}"
    time_periods_js_array = "[" + ", ".join([f"'{tp}'" for tp in time_periods_str]) + "]"
    slider_js = f"""
        <script type=\"text/javascript\">
            let timePeriods = {time_periods_js_array};
            let topicToColor = {topic_to_color_js};
            let sliderContainer = document.createElement(\"div\");
            sliderContainer.style.margin = \"10px\";
            sliderContainer.style.textAlign = \"center\";
            let timeLabel = document.createElement(\"div\");
            timeLabel.style.display = \"inline-block\";
            timeLabel.style.marginRight = \"10px\";
            timeLabel.innerHTML = \"Highlighting period: \" + timePeriods[0];
            let slider = document.createElement(\"input\");
            slider.type = \"range\";
            slider.min = 0;
            slider.max = timePeriods.length - 1;
            slider.value = 0;
            slider.id = \"timeSlider\";
            slider.style.width = \"70%\";
            slider.style.verticalAlign = \"middle\";
            slider.oninput = function() {{
                let selectedPeriod = timePeriods[this.value];
                timeLabel.innerHTML = \"Highlighting period: \" + selectedPeriod;
                network.body.data.edges.update(
                    network.body.data.edges.get().map(edge => {{
                        let isHighlighted = edge.time_period === selectedPeriod;
                        if (isHighlighted) {{
                            let topicColor = topicToColor[edge.topic] || '{default_edge_color}';
                            return {{
                                id: edge.id,
                                color: {{
                                    color: topicColor,
                                    highlight: topicColor,
                                    hover: topicColor
                                }},
                                width: 3
                            }};
                        }} else {{
                            return {{
                                id: edge.id,
                                color: {{
                                    color: '{default_edge_color}',
                                    highlight: '{default_edge_color}',
                                    hover: '{default_edge_color}'
                                }},
                                width: 1
                            }};
                        }}
                    }})
                );
            }};
            sliderContainer.appendChild(timeLabel);
            sliderContainer.appendChild(slider);
            let networkContainer = document.getElementById(\"mynetwork\");
            networkContainer.parentNode.insertBefore(sliderContainer, networkContainer);
        </script>
    """
    net.save_graph('temp_graph.html')
    with open('temp_graph.html', 'r') as f:
        html = f.read()
    html = html.replace('</body>', slider_js + '</body>')
    with open(output_file, 'w') as f:
        f.write(html)
    print(f"Interactive graph saved to {output_file}")

def main():
    # Always use both channels and groups tables for each country
    ru_tables = ['russian_channels_messages', 'russian_groups_messages']
    ua_tables = ['ukrainian_channels_messages', 'ukrainian_groups_messages']
    all_tables = ru_tables + ua_tables
    # Load all data once
    df_all = pd.concat([load_table_as_df(tbl) for tbl in all_tables], ignore_index=True)
    # Filter for Russian and Ukrainian networks
    df_ru = df_all[df_all['chat_name'].str.contains('ru', case=False, na=False) | df_all['chat_name'].str.contains('russian', case=False, na=False)].copy() if 'chat_name' in df_all.columns else df_all.copy()
    df_ua = df_all[df_all['chat_name'].str.contains('ua', case=False, na=False) | df_all['chat_name'].str.contains('ukrainian', case=False, na=False)].copy() if 'chat_name' in df_all.columns else df_all.copy()
    os.makedirs('network_graphs', exist_ok=True)
    visualize_forwarded_messages_with_highlight(df_ru, output_file='network_graphs/network_ru.html', time_unit='month')
    visualize_forwarded_messages_with_highlight(df_ua, output_file='network_graphs/network_ua.html', time_unit='month')
    visualize_forwarded_messages_with_highlight(df_all, output_file='network_graphs/network_all.html', time_unit='month')

if __name__ == "__main__":
    main() 
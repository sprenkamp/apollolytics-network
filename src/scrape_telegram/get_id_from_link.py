from telethon.sync import TelegramClient
from telethon.types import InputPeerChannel
from telethon.tl.functions import stats, messages
from telethon.tl.functions.channels import GetChannelsRequest
import datetime
import os
import json
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

TELEGRAM_API_ID = os.getenv('TELEGRAM_API_ID')
TELEGRAM_API_HASH = os.getenv('TELEGRAM_API_HASH')

def clean_chat_name(chat):
    link_components = chat.split('/')
    try:
        is_int = int(link_components[-1])
        chat = '/'.join(link_components[:-1])
    except ValueError:
        pass
    return chat

def get_chat_name(chat):
    link_components = chat.split('/')
    return link_components[3]

def get_link(chat, client):
    try:
        entity = client.get_entity(chat)
        id = entity.id
    except Exception as e:
        id = None
    return id

with TelegramClient('SessionName', TELEGRAM_API_ID, TELEGRAM_API_HASH) as client:
    df = pd.read_csv('./data/telegram/old_stuff_to_keep/telegram_links.csv')
    df['cleaned_link'] = df.LINKS.progress_apply(lambda x: clean_chat_name(x))
    df['chat_name'] = df.LINKS.progress_apply(lambda x: get_chat_name(x))
    df['id'] = df.chat_name.progress_apply(lambda x: get_link(x, client))
    df.to_csv('cleaned_channels_with_id.csv', index=False)
from telethon import TelegramClient, sync
from telethon.tl.functions import stats, messages
from telethon.types import PeerChannel
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.functions.messages import GetHistoryRequest
import numpy as np
import pandas as pd

TELEGRAM_API_ID = 24910635
TELEGRAM_API_HASH = '5583342def2592fe9e6cf13661d2da8f'

with TelegramClient('SessionName', TELEGRAM_API_ID, TELEGRAM_API_HASH) as client:
    df = pd.read_csv("../../data/telegram/old_stuff_to_keep/messages_scraped.csv")
    df = df[df.messageText != 'Невозможно подключиться к серверу. Проверьте соединение и повторите попытку.']
    df.describe()
    for i, row in df.iterrows():
        if not np.isnan(row['fwd_from']):
            try:
                entity = client.get_entity(PeerChannel(int(float(row['fwd_from']))))
                fullchn = client.invoke(GetFullChannelRequest(entity))
                print(fullchn)
                break
            except Exception as e:
                print(e)
                pass # No break as i forgot to add type of peer [user/channel] in dataset
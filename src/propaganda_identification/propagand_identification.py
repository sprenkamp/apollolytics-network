import asyncio
import websockets
import json
import time
from tqdm import tqdm
import pandas as pd

async def connect_to_websocket(response_dict, client_id, message, progress_bar, semaphore, timeout):
    async with semaphore:
        try:
            async with websockets.connect(
                "ws://13.48.71.178:8000/ws/analyze_propaganda", 
                ping_interval=None
            ) as websocket:
                request_data = {
                    "model_name": "gpt-4o-mini", 
                    "text": message,
                    "contextualize": "False"
                }

                await websocket.send(json.dumps(request_data))
                
                try:
                    # Wait for the response with a timeout
                    response = await asyncio.wait_for(websocket.recv(), timeout)
                    response = json.loads(response)  # Convert to JSON
                    
                    # Store the response data in the shared dictionary
                    response_dict[client_id] = response.get('data', None)
                except asyncio.TimeoutError:
                    # Timeout occurred, store an empty dictionary
                    response_dict[client_id] = {}
                except websockets.ConnectionClosedOK:
                    response_dict[client_id] = {}
                    pass
                except websockets.ConnectionClosedError:
                    response_dict[client_id] = {}
                    pass
                except Exception as e:
                    response_dict[client_id] = {}
        except Exception as e:
            print(f"Client {client_id}: An error occurred: {e}")
            response_dict[client_id] = {}
        finally:
            progress_bar.update(1)

async def simulate_multiple_clients(dataframe, parallel_connections, timeout):
    tasks = []
    semaphore = asyncio.Semaphore(parallel_connections)
    
    # Dictionary to collect responses
    response_dict = {}
    
    with tqdm(total=len(dataframe), desc="Finished Requests") as progress_bar:
        for i, row in enumerate(dataframe.itertuples()):
            client_id = row.Index
            tasks.append(
                connect_to_websocket(response_dict, client_id, row.messageText, progress_bar, semaphore, timeout)
            )
        
        await asyncio.gather(*tasks)
    
    return response_dict

if __name__ == "__main__":
    # ADJUSTABLE PARAMETERS
    PARALLEL_CONNECTIONS = 50
    TIMEOUT = 150  # Timeout in seconds for each request
    INPUT_FILE = "data/telegram/messages_scraped.csv"  # Path to your input CSV file
    SORT_AFTER = "forwards"  # Column to sort the DataFrame by
    NUM_TOP_MESSAGES = 10000  # Number of top messages to select

    # Load your DataFrame
    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values(by=SORT_AFTER, ascending=False)
    df = df.head(NUM_TOP_MESSAGES)

    # Run the async tasks and get the responses
    response_dict = asyncio.run(simulate_multiple_clients(df, PARALLEL_CONNECTIONS, TIMEOUT))

    # Add the results to the DataFrame
    df['response'] = df.index.map(response_dict)

    # Save the updated DataFrame to a CSV file
    df.to_csv(INPUT_FILE.split(".")[0] + "_" + "propaganda" + "_" + SORT_AFTER + str(NUM_TOP_MESSAGES) + ".csv",
              index=False)
